"""SGLang tap: one UDP datagram per scheduler step, plus a KV memory map once a second.

SGLang loads it through the `sglang.srt.plugins` entry point in every process. It adds an AFTER
hook on `Scheduler.process_batch_result` and reads CPU-side fields only (no GPU syncs).

Only the reporting rank sends: pp_rank 0, attention-TP rank 0, attention-CP rank 0, the same rule
SGLang uses for its own metrics. Tensor-parallel ranks on other GPUs or nodes stay silent, so a
multi-node engine reports once, from node 0. With data parallelism, each DP rank reports as its
own instance (`<instance>.dp<N>`).

Tested with SGLang 0.5.19 and 0.5.23.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from typing import Any

from . import UdpSender

_tx: UdpSender | None = None
_seq = 0
_hello_sent = False
_reported: "OrderedDict[str, None]" = OrderedDict()


def _g(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _int(x: Any) -> int | None:
    return x if isinstance(x, int) else None


def _parallel(sched: Any) -> dict[str, int]:
    """This process's ranks. SGLang keeps them in runtime_context (0.5.19+) and on Scheduler.ps."""
    ctx = None
    try:
        from sglang.srt.runtime_context import get_parallel
        ctx = get_parallel()
    except Exception:
        ctx = None
    ps = _g(sched, "ps")
    out: dict[str, int] = {}
    for name in ("pp_rank", "attn_tp_rank", "attn_cp_rank", "tp_rank", "dp_rank"):
        v = None
        for src in (ctx, ps, sched):
            try:
                v = getattr(src, name) if src is not None else None
            except Exception:
                v = None
            if v is not None:
                break
        out[name] = int(v) if isinstance(v, int) else 0
    return out


def _hello(sched: Any, instance: str, ranks: dict[str, int]) -> None:
    sa = _g(sched, "server_args")
    try:
        import sglang
        version = sglang.__version__
    except Exception:
        version = None
    cfg = {
        "model": _g(sa, "model_path"),
        "max_running_requests": _g(sa, "max_running_requests") or _g(sched, "max_running_requests"),
        "chunked_prefill_size": _g(sa, "chunked_prefill_size") or _g(sched, "chunked_prefill_size"),
        "max_prefill_tokens": _g(sa, "max_prefill_tokens"),
        "schedule_policy": _g(sa, "schedule_policy"),
        "page_size": _g(sa, "page_size") or _g(sched, "page_size"),
        "mem_fraction_static": _g(sa, "mem_fraction_static"),
        "kv_cache_dtype": _g(sa, "kv_cache_dtype"),
        "prefix_caching": not _g(sa, "disable_radix_cache", False),
        "overlap_schedule": not _g(sa, "disable_overlap_schedule", False),
        "tp": _g(sa, "tp_size"), "dp": _g(sa, "dp_size"), "ep": _g(sa, "ep_size"), "pp": _g(sa, "pp_size"),
        "nnodes": _g(sa, "nnodes"), "node_rank": _g(sa, "node_rank"), "reporting_rank": ranks,
        "spec_method": _g(sa, "speculative_algorithm"),
        "spec_tokens": _g(sa, "speculative_num_draft_tokens"),
        "disaggregation_mode": _g(sa, "disaggregation_mode"),
        "max_total_tokens": _g(sched, "max_total_num_tokens"),
    }
    _tx.send({"type": "tap.hello", "engine": "sglang", "engine_version": version, "instance": instance,
              "config": cfg, "url": _tx.url})


def _reason(r: Any) -> str | None:
    if r is None:
        return None
    try:
        j = r.to_json()  # SGLang FINISH_* objects
        return j.get("type") if isinstance(j, dict) else str(j)
    except Exception:
        return type(r).__name__.removeprefix("FINISH_").lower()


def _finished(batch: Any) -> list[dict[str, Any]]:
    out = []
    for req in _g(batch, "reqs", []) or []:
        try:
            if not req.finished():
                continue
        except Exception:
            continue
        rid = str(_g(req, "rid", ""))
        if not rid or rid in _reported:
            continue
        _reported[rid] = None
        if len(_reported) > 50_000:
            _reported.popitem(last=False)
        ts = _g(req, "time_stats")
        wq, fe = _g(ts, "wait_queue_entry_time", 0.0), _g(ts, "forward_entry_time", 0.0)
        pf, done = _g(ts, "prefill_finished_time", 0.0), _g(ts, "completion_time", 0.0)
        out.append({
            "request_id": rid,
            "queued_s": max(0.0, fe - wq) if fe and wq else None,
            "prefill_s": max(0.0, pf - fe) if pf and fe else None,
            "decode_s": max(0.0, done - pf) if done and pf else None,
            "prompt_tokens": len(_g(req, "origin_input_ids", []) or []),
            "cached_tokens": _int(_g(req, "cached_tokens")) or 0,
            "gen_tokens": len(_g(req, "output_ids", []) or []),
            "preemptions": _int(_g(req, "retraction_count")) or 0,
            "finish_reason": _reason(_g(req, "finished_reason")),
        })
    return out


def on_result(ret: Any, sched: Any, batch: Any = None, result: Any = None, *args: Any, **kwargs: Any) -> None:
    """AFTER hook on Scheduler.process_batch_result(self, batch, result)."""
    global _seq, _hello_sent
    try:
        ranks = _parallel(sched)
        if ranks["pp_rank"] or ranks["attn_tp_rank"] or ranks["attn_cp_rank"]:
            return
        dp_size = _g(_g(sched, "server_args"), "dp_size", 1) or 1
        instance = f"{_tx.instance}.dp{ranks['dp_rank']}" if dp_size > 1 else _tx.instance
        if not _hello_sent:
            _hello(sched, instance, ranks)
            _hello_sent = True
        mode = _g(batch, "forward_mode")
        if mode is None or (hasattr(mode, "is_idle") and mode.is_idle()):
            return
        n = len(_g(batch, "reqs", []) or [])
        is_extend = bool(mode.is_extend()) if hasattr(mode, "is_extend") else False
        is_mixed = bool(mode.is_mixed()) if hasattr(mode, "is_mixed") else False
        decoding = _g(batch, "decoding_reqs", None)
        gen_reqs = (len(decoding) if decoding else 0) if is_mixed else (0 if is_extend else n)
        ctx_tokens = (_int(_g(batch, "extend_num_tokens")) or 0) if (is_extend or is_mixed) else 0
        extra = _int(_g(result, "num_correct_drafts"))
        launch = _g(batch, "launch_ts")
        msg: dict[str, Any] = {
            "type": "engine.step", "engine": "sglang", "instance": instance, "seq": _seq,
            "running": len(_g(_g(sched, "running_batch"), "reqs", []) or []),
            "waiting": len(_g(sched, "waiting_queue", []) or []),
            "iter": {
                "i": _g(batch, "forward_iter"),
                "ctx_reqs": (n - gen_reqs) if (is_extend or is_mixed) else 0,
                "ctx_tokens": ctx_tokens,
                "gen_reqs": gen_reqs,
                "gen_tokens": gen_reqs + (extra or 0),
                "elapsed_ms": (time.monotonic() - launch) * 1000 if isinstance(launch, float) else None,
                "dummy": False,
                "cuda_graph": _g(result, "can_run_cuda_graph"),
            },
            "gen_tokens": gen_reqs + (extra or 0),
        }
        _seq += 1
        pso = _g(sched, "pool_stats_observer")
        if pso is not None:
            try:
                ps = pso.get_pool_stats()
                msg["kv_usage"] = _g(ps, "full_token_usage")
            except Exception:
                pass
        stats = _g(batch, "prefill_stats")
        if stats is not None:
            new, hit = _int(_g(stats, "log_input_tokens")), _int(_g(stats, "log_hit_tokens"))
            if new is not None and hit is not None and new + hit > 0:
                msg["prefix"] = {"queries": new + hit, "hits": hit}
        fin = _finished(batch)
        if fin:
            msg["finished"] = fin
        msg["batch"] = _batch_members(batch, is_extend, is_mixed, decoding, extra)
        _tx.send(msg)
        _kv_map_tick(sched, instance)
    except Exception:
        pass  # a tap must never break the engine
    return None


def _batch_members(batch: Any, is_extend: bool, is_mixed: bool, decoding: Any, extra: int | None) -> dict[str, Any]:
    """Who was in this forward pass: [rid, phase, tokens] per request. CPU-side fields only."""
    reqs = _g(batch, "reqs", []) or []
    ext = _g(batch, "extend_lens", None) or []
    dec_ids = {id(r) for r in (decoding or [])} if is_mixed else set()
    out = []
    for i, r in enumerate(reqs[:BATCH_MAX]):
        rid = str(_g(r, "rid", ""))
        if is_extend and id(r) not in dec_ids:
            toks = ext[i] if i < len(ext) and isinstance(ext[i], int) else _int(_g(r, "extend_input_len")) or 0
            out.append([rid, "p", toks])
        else:
            out.append([rid, "d", 1])
    return {"reqs": out, "truncated": len(reqs) > BATCH_MAX}


# ---- KV memory map ---------------------------------------------------------------------------
# Every KV_MAP_EVERY seconds: which KV slots each running request holds, and which are free, binned
# for drawing. GPU reads go on a side stream with a non-blocking copy, and are turned into a message on
# a later tick once the copy has landed, so the scheduler thread never waits on the GPU.

BATCH_MAX = 512
KV_MAP_EVERY = float(os.environ.get("IO_KV_MAP_EVERY", "1.0"))
KV_MAP_BINS = 4096
_kv_last = 0.0
_kv_pending: dict[str, Any] | None = None
_kv_stream = None


def _ranges(sorted_bins: list[int]) -> list[list[int]]:
    """[1,2,3,7,8] -> [[1,4],[7,9]] (half-open)."""
    out: list[list[int]] = []
    for b in sorted_bins:
        if out and out[-1][1] == b:
            out[-1][1] = b + 1
        else:
            out.append([b, b + 1])
    return out


def _kv_map_tick(sched: Any, instance: str) -> None:
    global _kv_last, _kv_pending, _kv_stream
    if KV_MAP_EVERY <= 0:
        return
    if _kv_pending is not None:
        if not _kv_pending["event"].query():
            return
        _kv_emit(_kv_pending, instance)
        _kv_pending = None
    now = time.monotonic()
    if now - _kv_last < KV_MAP_EVERY:
        return
    _kv_last = now
    import torch

    pool = _g(_g(sched, "req_to_token_pool"), "req_to_token")
    alloc = _g(sched, "token_to_kv_pool_allocator")
    if pool is None or alloc is None:
        return
    page = int(_g(alloc, "page_size", 1) or 1)
    cap = int(_g(alloc, "size", 0) or 0)  # KV capacity in tokens (slot 0 is reserved padding)
    if cap <= 0:
        return
    # SGLang 0.5.23 keeps a request's KV bookkeeping in req.kv (ReqKvInfo): req_pool_idx is its row in
    # req_to_token, kv_committed_len how many tokens of KV have actually been written.
    rids, rows, lens, prefix = [], [], [], []
    width = pool.shape[1]
    seen = set()
    for r in (_g(_g(sched, "running_batch"), "reqs", []) or []):
        kvi = _g(r, "kv")
        idx = _g(kvi, "req_pool_idx", _g(r, "req_pool_idx"))
        rid = str(_g(r, "rid", ""))
        if idx is None or rid in seen:
            continue
        seen.add(rid)
        n = _g(kvi, "kv_committed_len", 0) or _g(kvi, "kv_allocated_len", 0) or int(r.seqlen)
        n = min(int(n), width)
        if n > 0:
            rids.append(rid)
            rows.append(int(idx))
            lens.append(n)
            # [0, cache_protected_len) is owned by the radix tree: prefix reused from (or published to)
            # the cache. Pages past it are this request's own.
            prefix.append(min(n, int(_g(kvi, "cache_protected_len", 0) or 0)))
    bin_tokens = -(-(cap + 1) // KV_MAP_BINS)
    if _kv_stream is None:
        _kv_stream = torch.cuda.Stream(device=pool.device)
    with torch.cuda.stream(_kv_stream):
        parts = [pool[row, :n] for row, n in zip(rows, lens)]
        owned = (torch.cat(parts).to(torch.int64) // bin_tokens) if parts else torch.empty(0, dtype=torch.int64, device=pool.device)
        frees = [t for t in (_g(alloc, "free_pages"), _g(alloc, "release_pages")) if t is not None and t.numel()]
        free_tok = (torch.cat(frees).to(torch.int64) * page) if frees else torch.empty(0, dtype=torch.int64, device=pool.device)
        free_count = torch.bincount(free_tok // bin_tokens, minlength=KV_MAP_BINS)[:KV_MAP_BINS] * page
        owned_cpu = owned.to("cpu", non_blocking=True)
        free_cpu = free_count.to("cpu", non_blocking=True)
        event = torch.cuda.Event()
        event.record(_kv_stream)
    _kv_pending = {"event": event, "rids": rids, "lens": lens, "prefix": prefix, "owned": owned_cpu, "free": free_cpu,
                   "cap": cap, "bin_tokens": bin_tokens, "t_ns": time.time_ns()}


def _kv_emit(p: dict[str, Any], instance: str) -> None:
    owned = p["owned"].tolist()
    free_count = p["free"].tolist()
    per_req, off = [], 0
    for n, pre in zip(p["lens"], p["prefix"]):
        seg = owned[off: off + n]
        per_req.append((set(seg[:pre]), set(seg[pre:])))
        off += n
    # Coarsen until the datagram fits (heavy fragmentation means many ranges).
    for f in (1, 2, 4, 8, 16):
        bins = KV_MAP_BINS // f
        bt = p["bin_tokens"] * f
        owners, owned_bins, prefix_count = [], set(), {}
        for rid, n, pre, (pset, oset) in zip(p["rids"], p["lens"], p["prefix"], per_req):
            pb = sorted({b // f for b in pset})
            ob = sorted({b // f for b in oset})
            owned_bins.update(pb)
            owned_bins.update(ob)
            for b in pb:
                prefix_count[b] = prefix_count.get(b, 0) + 1
            owners.append({"rid": rid, "tokens": n, "prefix_tokens": pre,
                           "prefix_ranges": _ranges(pb), "own_ranges": _ranges(ob),
                           "ranges": _ranges(sorted(set(pb) | set(ob)))})
        # True sharing: prefix pages referenced by two or more running requests at once.
        shared_bins = sorted(b for b, c in prefix_count.items() if c >= 2)
        fc = [sum(free_count[i * f:(i + 1) * f]) for i in range(bins)]
        free_bins = [b for b, c in enumerate(fc) if c >= bt * 0.5]
        free_set = set(free_bins)
        cached_bins = [b for b in range(bins) if b not in free_set and b not in owned_bins]
        msg = {"type": "kv.map", "engine": "sglang", "instance": instance, "t_ns": p["t_ns"],
               "capacity_tokens": p["cap"], "bin_tokens": bt, "bins": bins,
               "used_tokens": p["cap"] - int(sum(free_count)), "owned_tokens": int(sum(p["lens"])),
               "owners": owners, "shared": _ranges(shared_bins),
               "free": _ranges(free_bins), "cached": _ranges(cached_bins)}
        if len(json.dumps(msg, separators=(",", ":"))) < 60_000:
            break
    _tx.send(msg)


def register() -> None:
    global _tx
    _tx = UdpSender("sglang-tap")
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.scheduler.Scheduler.process_batch_result", on_result, HookType.AFTER)
