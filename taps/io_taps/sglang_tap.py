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
_hello_sent = -1e9  # monotonic time of the last tap.hello
HELLO_EVERY = 30.0
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


_out_len: "OrderedDict[str, int]" = OrderedDict()  # rid -> output tokens seen so far
_prefilled: "OrderedDict[str, None]" = OrderedDict()  # requests whose prefix lookup was counted


def _prefix_once(batch: Any, decoding: Any) -> tuple[int, int] | None:
    """Prefix-cache lookups, each request once at its first prefill chunk: prompt tokens, and how many of them it
    found in the cache (SGLang's per-request count, the one the API reports). Per-chunk counters would count a
    chunked prompt's earlier chunks as hits again."""
    dec = {id(r) for r in (decoding or [])}
    q = h = 0
    for r in _g(batch, "reqs", []) or []:
        rid = str(_g(r, "rid", "") or "")
        if id(r) in dec or not rid or rid in _prefilled:
            continue
        _prefilled[rid] = None
        q += len(_g(r, "origin_input_ids", []) or [])
        h += int(_g(r, "cached_tokens", 0) or 0)
    while len(_prefilled) > 50_000:
        _prefilled.popitem(last=False)
    return (q, h) if q else None


def _generated(reqs: list[Any]) -> int | None:
    """Tokens appended to requests' outputs in this step (first tokens and accepted drafts included)."""
    total, seen = 0, False
    for r in reqs:
        out = _g(r, "output_ids")
        if out is None:
            continue
        seen = True
        rid = str(_g(r, "rid", ""))
        n = len(out)
        total += max(0, n - _out_len.get(rid, 0))
        # finished requests stay in the table: with overlap scheduling they can appear in the next batch too
        _out_len[rid] = n
        _out_len.move_to_end(rid)
    while len(_out_len) > 20_000:
        _out_len.popitem(last=False)
    return total if seen else None


def on_result(ret: Any, sched: Any, batch: Any = None, result: Any = None, *args: Any, **kwargs: Any) -> None:
    """AFTER hook on Scheduler.process_batch_result(self, batch, result)."""
    global _seq, _hello_sent
    try:
        ranks = _parallel(sched)
        if ranks["pp_rank"] or ranks["attn_tp_rank"] or ranks["attn_cp_rank"]:
            return
        dp_size = _g(_g(sched, "server_args"), "dp_size", 1) or 1
        instance = f"{_tx.instance}.dp{ranks['dp_rank']}" if dp_size > 1 else _tx.instance
        if time.monotonic() - _hello_sent > HELLO_EVERY:  # repeated, so a restarted collector learns the config
            _hello(sched, instance, ranks)
            _hello_sent = time.monotonic()
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
        gen = _generated(_g(batch, "reqs", []) or [])
        if gen is None:
            gen = gen_reqs + (extra or 0)
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
                "gen_tokens": gen,
                "elapsed_ms": (time.monotonic() - launch) * 1000 if isinstance(launch, float) else None,
                "dummy": False,
                "cuda_graph": _g(result, "can_run_cuda_graph"),
            },
            "gen_tokens": gen,
        }
        _seq += 1
        pso = _g(sched, "pool_stats_observer")
        if pso is not None:
            try:
                ps = pso.get_pool_stats()
                msg["kv_usage"] = _g(ps, "full_token_usage")
            except Exception:
                pass
        if is_extend or is_mixed:
            pre = _prefix_once(batch, decoding if is_mixed else None)
            if pre:
                msg["prefix"] = {"queries": pre[0], "hits": pre[1]}
        fin = _finished(batch)
        if fin:
            msg["finished"] = fin
        msg["batch"] = _batch_members(batch, is_extend, is_mixed, decoding, extra)
        _tx.send(msg)
        _kv_map_tick(sched, instance, batch)
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
# Every KV_MAP_EVERY seconds, every KV slot is classified exactly:
#   shared  referenced by two or more running requests (counted once)
#   reused  referenced by one running request, inside the prefix it found in the cache at admission
#   own     referenced by one running request, computed or generated by it
#   cached  allocated, referenced by no running request (the radix tree keeps it for reuse)
#   free    on the allocator's free list
# The message carries exact token totals, per-cell token counts (as levels of the cell size, for
# drawing), and per request: tokens, cached tokens, shared tokens, its sharing group and the cells it
# touches. GPU work runs on a side stream with a non-blocking copy and is turned into a message on a
# later tick once the copy has landed, so the scheduler thread never waits on the GPU.

BATCH_MAX = 512
KV_MAP_EVERY = float(os.environ.get("IO_KV_MAP_EVERY", "1.0"))
KV_MAP_BINS = 4096
KV_MAP_BUDGET = 60_000  # bytes per datagram
_kv_last = 0.0
_kv_pending: dict[str, Any] | None = None
_kv_stream = None


def _kv_reqs(sched: Any, batch: Any) -> list[Any]:
    """Requests that hold KV right now: the running batch, the batch whose results are being processed, the
    batches already launched behind it (overlap scheduling keeps them in result_queue), the chunked prefill."""
    out, seen = [], set()
    launched = [_g(b, "reqs", []) or [] for b, *_ in list(_g(sched, "result_queue", None) or [])]
    for group in (_g(_g(sched, "running_batch"), "reqs", []) or [], _g(batch, "reqs", []) or [], *launched,
                  [_g(sched, "chunked_req")]):
        for r in group:
            rid = str(_g(r, "rid", "") or "")
            if not rid or rid in seen:
                continue
            fin = _g(r, "finished")
            try:
                if callable(fin) and fin():
                    continue
            except Exception:
                pass
            seen.add(rid)
            out.append(r)
    return out


def _kv_map_tick(sched: Any, instance: str, batch: Any = None) -> None:
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
    cap = int(_g(alloc, "size", 0) or 0)  # KV capacity in tokens; slot 0 is reserved padding
    if cap <= 0:
        return
    width = pool.shape[1]
    rids, rows, lens, cached, protected = [], [], [], [], []
    for r in _kv_reqs(sched, batch):
        kvi = _g(r, "kv")
        idx = _g(kvi, "req_pool_idx", _g(r, "req_pool_idx"))
        if idx is None:
            continue
        n = _g(kvi, "kv_allocated_len", 0) or _g(kvi, "kv_committed_len", 0) or int(r.seqlen)  # slots it holds
        n = min(int(n), width)
        if n <= 0:
            continue
        rids.append(str(r.rid))
        rows.append(int(idx))
        lens.append(n)
        cached.append(min(n, int(_g(r, "cached_tokens", 0) or 0)))
        protected.append(min(n, int(_g(kvi, "cache_protected_len", 0) or 0)))
    # the radix tree's own accounting at this instant, so readers can check the map against the engine
    tc = _g(sched, "tree_cache")
    sizes: dict[str, int | None] = {}
    for k in ("evictable", "protected"):
        f = _g(tc, f"{k}_size")
        try:
            sizes[k] = int(f()) if callable(f) else None
        except Exception:
            sizes[k] = None
    bins = KV_MAP_BINS
    bt = -(-(cap + 1) // bins)
    if _kv_stream is None:
        _kv_stream = torch.cuda.Stream(device=pool.device)
    _kv_stream.wait_stream(torch.cuda.current_stream(pool.device))
    with torch.cuda.stream(_kv_stream):
        dev = pool.device
        nreq = len(rows)
        zeros = torch.zeros(bins, dtype=torch.int64, device=dev)
        own_c = reused_c = shared_c = zeros
        per_req = torch.zeros((3, max(1, nreq)), dtype=torch.int64, device=dev)
        touch = torch.zeros((max(1, nreq), bins), dtype=torch.bool, device=dev)
        if nreq:
            slots = torch.cat([pool[row, :n] for row, n in zip(rows, lens)]).to(torch.int64)
            lens_t = torch.tensor(lens, device=dev)
            req_i = torch.repeat_interleave(torch.arange(nreq, device=dev), lens_t)
            starts = torch.cumsum(lens_t, 0) - lens_t
            pos = torch.arange(slots.numel(), device=dev) - starts[req_i]
            reused = (pos < torch.tensor(cached, device=dev)[req_i]).to(torch.int64)
            # only the tree-owned part [0, cache_protected_len) can be shared; the rest is the request's own
            tree = (pos < torch.tensor(protected, device=dev)[req_i]).to(torch.int64)
            uniq, inv = torch.unique(slots, return_inverse=True)  # every physical slot counted once
            nu = uniq.numel()
            multi = torch.zeros(nu, dtype=torch.int64, device=dev).scatter_add_(0, inv, tree) > 1
            reused_u = torch.zeros(nu, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, reused, "amax")
            ubin = (uniq // bt).clamp_(max=bins - 1)
            single = ~multi
            shared_c = torch.bincount(ubin[multi], minlength=bins)[:bins]
            reused_c = torch.bincount(ubin[single & (reused_u > 0)], minlength=bins)[:bins]
            own_c = torch.bincount(ubin[single & (reused_u == 0)], minlength=bins)[:bins]
            shared_slot = multi[inv] & (tree > 0)
            # sharing groups: each shared slot belongs to the lowest request index that holds it in its prefix
            ref = torch.where(tree > 0, req_i, torch.full_like(req_i, nreq))
            leader_u = torch.full((nu,), nreq, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, ref, "amin")
            leader = leader_u[inv]
            per_req[0] = torch.bincount(req_i[shared_slot], minlength=nreq)[:nreq]
            per_req[1] = torch.full((nreq,), nreq, dtype=torch.int64, device=dev).scatter_reduce_(
                0, req_i[shared_slot], leader[shared_slot], "amin")
            per_req[2] = torch.bincount(leader_u[multi], minlength=nreq)[:nreq]
            touch[req_i, (slots // bt).clamp_(max=bins - 1)] = True
        frees = [t for t in (_g(alloc, "free_pages"), _g(alloc, "release_pages")) if t is not None and t.numel()]
        for t in frees:
            t.record_stream(_kv_stream)  # the allocator may replace these tensors while this stream reads them
        free_tok = (torch.cat(frees).to(torch.int64) * page) if frees else torch.empty(0, dtype=torch.int64, device=dev)
        free_c = torch.bincount((free_tok // bt).clamp_(max=bins - 1), minlength=bins)[:bins] * page
        counts = torch.stack([own_c, reused_c, shared_c, free_c]).to("cpu", non_blocking=True)
        per_req_cpu = per_req.to("cpu", non_blocking=True)
        touch_cpu = touch.to("cpu", non_blocking=True)
        event = torch.cuda.Event()
        event.record(_kv_stream)
    _kv_pending = {"event": event, "rids": rids, "lens": lens, "cached": cached, "protected": protected,
                   "counts": counts, "per_req": per_req_cpu, "touch": touch_cpu, "cap": cap, "bin_tokens": bt,
                   "tree": sizes, "t_ns": time.time_ns()}


def _b64(a: Any) -> str:
    import base64

    return base64.b64encode(a.tobytes()).decode()


def _kv_emit(p: dict[str, Any], instance: str) -> None:
    import numpy as np

    counts = p["counts"].numpy()  # own, reused, shared, free per cell
    cap, bt, nreq = p["cap"], p["bin_tokens"], len(p["rids"])
    bins = counts.shape[1]
    size = np.full(bins, bt, dtype=np.int64)
    size[-1] = (cap + 1) - bt * (bins - 1)  # slots 0..cap
    size[0] -= 1  # slot 0 is reserved padding
    cells = np.vstack([counts, np.zeros((1, bins), dtype=np.int64)])
    cells[4] = np.maximum(0, size - counts.sum(axis=0))  # cached: allocated, no running request
    own, reused, shared, free = (int(x) for x in counts.sum(axis=1))
    totals = {"own": own, "reused": reused, "shared": shared, "free": free,
              "cached": max(0, cap - own - reused - shared - free)}
    per_req = p["per_req"].numpy()
    touch = p["touch"].numpy()
    for f in (1, 2, 4, 8, 16):
        nb = bins // f
        csize = size.reshape(nb, f).sum(axis=1)
        lv = np.rint(255 * cells.reshape(5, nb, f).sum(axis=2) / np.maximum(1, csize)).clip(0, 255).astype(np.uint8)
        tb = touch[:nreq].reshape(nreq, nb, f).any(axis=2) if nreq else touch[:0]
        owners = []
        for i, rid in enumerate(p["rids"]):
            g = int(per_req[1, i])
            owners.append({"rid": rid, "tokens": p["lens"][i], "cached_tokens": p["cached"][i],
                           "prefix_tokens": p["protected"][i], "shared_tokens": int(per_req[0, i]),
                           "group": p["rids"][g] if g < nreq else None,
                           "group_tokens": int(per_req[2, g]) if g < nreq else 0,
                           "cells": _b64(np.packbits(tb[i]))})
        msg = {"type": "kv.map", "engine": "sglang", "instance": instance, "t_ns": p["t_ns"], "format": 2,
               "capacity_tokens": cap, "bin_tokens": bt * f, "bins": nb, "tokens": totals,
               "used_tokens": cap - free, "owned_tokens": own + reused + shared, "tree": p["tree"],
               "cells": {k: _b64(lv[j]) for j, k in enumerate(("own", "reused", "shared", "free", "cached"))},
               "owners": owners}
        if len(json.dumps(msg, separators=(",", ":"))) < KV_MAP_BUDGET:
            break
    _tx.send(msg)


def register() -> None:
    global _tx
    _tx = UdpSender("sglang-tap")
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.scheduler.Scheduler.process_batch_result", on_result, HookType.AFTER)
