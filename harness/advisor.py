"""Advisor: watches the live stream and posts evidence-backed proposals to the board.

Each detector reads a sliding window of signals, and when its condition holds it posts an
`advisor.note` with the evidence, a concrete config change, and a predicted effect. Notes are
keyed by detector, so a persisting condition updates its note instead of spamming. A proposal
carries an `experiment` block that the experiment runner can execute ("test it").

This layer is deterministic on purpose: it is the part that must be right. An LLM layer can sit
on top later to narrate, combine detectors and plan multi-step experiments.

  io-advisor [--collector ws://127.0.0.1:18200/ws] [--post http://127.0.0.1:18200/note]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from aiohttp import ClientSession, WSMsgType

from . import sse

WINDOW_S = 30.0
EVAL_EVERY_S = 5.0


def q(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


@dataclass
class State:
    hello: dict[str, Any] = field(default_factory=dict)
    arrive: dict[str, int] = field(default_factory=dict)
    first: dict[str, int] = field(default_factory=dict)
    workload: dict[str, str] = field(default_factory=dict)
    # (t_mono, ttft_s, queued_s, prefill_s, workload)
    ttft: deque = field(default_factory=lambda: deque(maxlen=5000))
    steps: deque = field(default_factory=lambda: deque(maxlen=20000))   # (t_mono, step msg)
    ticks: deque = field(default_factory=lambda: deque(maxlen=600))     # (t_mono, engine.tick)
    gpu: deque = field(default_factory=lambda: deque(maxlen=600))       # (t_mono, gpu.tick)
    family: dict = field(default_factory=dict)                          # rid -> prefix family
    # (t_mono, family, workload, prompt_tokens, cached_tokens) from req.prefill_stats
    prefill: deque = field(default_factory=lambda: deque(maxlen=5000))
    moe: deque = field(default_factory=lambda: deque(maxlen=120))       # (t_mono, moe.tick)

    def recent(self, dq: deque, seconds: float = WINDOW_S) -> list[Any]:
        now = time.monotonic()
        return [x[1:] if len(x) > 2 else x[1] for x in dq if now - x[0] <= seconds]

    def config(self) -> dict[str, Any]:
        cfg = dict(self.hello.get("config") or {})
        for inst in (self.hello.get("instances") or {}).values():
            for k, v in (inst.get("config") or {}).items():
                cfg.setdefault(k, v)
        return cfg


Note = dict[str, Any]
Detector = Callable[[State], Note | None]

# The same knob under each engine's flag name (None: no equivalent, so no experiment is proposed).
FLAGS: dict[str, dict[str, Callable[[Any], list[str]] | None]] = {
    "max_seqs": {"vllm": lambda v: ["--max-num-seqs", str(v)], "sglang": lambda v: ["--max-running-requests", str(v)]},
    "token_budget": {"vllm": lambda v: ["--max-num-batched-tokens", str(v)],
                     "sglang": lambda v: ["--chunked-prefill-size", str(v)]},
    "kv_fp8": {"vllm": lambda v: ["--kv-cache-dtype", "fp8"], "sglang": lambda v: ["--kv-cache-dtype", "fp8_e5m2"]},
    "api_servers": {"vllm": lambda v: ["--api-server-count", str(v)], "sglang": None},
    "prefix_cache": {"vllm": lambda v: ["--enable-prefix-caching"], "sglang": None},  # on by default in SGLang
    "eplb": {"vllm": lambda v: ["--enable-eplb"], "sglang": lambda v: ["--enable-eplb"]},
}
CONFIG_KEYS = {"max_seqs": {"vllm": "max_num_seqs", "sglang": "max_running_requests"},
               "token_budget": {"vllm": "max_num_batched_tokens", "sglang": "chunked_prefill_size"}}


def engine_of(s: "State") -> str:
    return s.hello.get("engine") or "vllm"


def flag(s: "State", knob: str, value: Any = None) -> list[str] | None:
    f = FLAGS.get(knob, {}).get(engine_of(s))
    return f(value) if f else None


def cfg_value(s: "State", knob: str, default: Any = None) -> Any:
    key = CONFIG_KEYS.get(knob, {}).get(engine_of(s))
    return s.config().get(key) if key else default


def experiment(s: "State", knob: str, value: Any, bench: list[str], goal: str) -> dict[str, Any] | None:
    args = flag(s, knob, value)
    return {"args": args, "bench": bench, "goal": goal} if args else None


def frontend_bound(s: State) -> Note | None:
    rows = s.recent(s.ttft)
    if len(rows) < 20:
        return None
    ttft = q([r[0] for r in rows], 0.5)
    outside = q([r[0] - r[1] - r[2] for r in rows], 0.5)
    engine = q([r[1] + r[2] for r in rows], 0.5)
    # Fire when more of TTFT is spent outside the engine than inside it, and it is material.
    if not ttft or outside is None or engine is None or outside < 0.1 or outside < engine:
        return None
    cur = int(s.config().get("api_server_count") or 1)
    nxt = min(8, max(2, cur * 2))
    return {
        "agent": "advisor · frontend", "title": f"Frontend-bound: {outside / ttft:.0%} of TTFT is outside the engine",
        "body": f"Over the last {WINDOW_S:.0f}s ({len(rows)} requests) TTFT p50 is {ttft * 1000:.0f} ms, of which "
                f"{outside * 1000:.0f} ms is spent before the engine queues the request or after it emits the token "
                f"(tokenize, detokenize, HTTP streaming), versus {engine * 1000:.0f} ms inside the engine. "
                f"The API server process is the bottleneck, not the GPU.",
        "predicted": f"{nxt} API server processes: TTFT p50 toward ~{max(engine * 1000 + 30, 50):.0f} ms",
        "experiment": experiment(s, "api_servers", nxt, ["burst", "rag"], "ttft_p90"),
    }


def queue_bound(s: State) -> Note | None:
    steps = s.recent(s.steps, 15)
    if len(steps) < 20:
        return None
    # p75 rather than the median: the load loop alternates bursts with lighter traffic.
    waiting = q([m.get("waiting") or 0 for m in steps], 0.75) or 0
    if waiting < 5:
        return None
    kv = q([m.get("kv_usage") or 0 for m in steps], 0.5) or 0
    running = q([m.get("running") or 0 for m in steps], 0.9) or 0
    max_seqs = cfg_value(s, "max_seqs") or 0
    if kv > 0.9:
        return {
            "agent": "advisor · memory", "title": f"Requests queue because KV cache is full ({kv:.0%})",
            "body": f"Median {waiting:.0f} requests waiting with KV usage at {kv:.0%} and {running:.0f} running. "
                    "The scheduler cannot admit more sequences without evicting or preempting.",
            "predicted": "FP8 KV cache roughly doubles KV capacity: waiting → ~0, TTFT p90 down",
            "experiment": experiment(s, "kv_fp8", None, ["burst"], "ttft_p90"),
        }
    if max_seqs and running >= 0.95 * max_seqs:
        # Size the new limit to measured demand, not a blind doubling.
        demand = max((m.get("running") or 0) + (m.get("waiting") or 0) for m in steps)
        target = max(max_seqs * 2, 1 << max(0, int(demand * 1.25) - 1).bit_length())
        return {
            "agent": "advisor · scheduler", "title": f"Requests queue at the sequence limit ({max_seqs})",
            "body": f"{running:.0f} requests running at the engine's limit of {max_seqs}, {waiting:.0f} waiting "
                    f"(p75 over 15 s), yet KV is only {kv:.1%} used — memory has room, the scheduler cap does not.",
            "predicted": f"Running-request limit {target} (peak demand {demand}): waiting → ~0 while KV has room; "
                         "burst TTFT p90 back toward the unconstrained baseline",
            "experiment": experiment(s, "max_seqs", target, ["burst"], "ttft_p90"),
        }
    return None


def prefill_interference(s: State) -> Note | None:
    steps = [m for m in s.recent(s.steps) if m.get("iter") and not m["iter"].get("dummy")]
    mixed = [m["iter"]["elapsed_ms"] for m in steps if m["iter"]["ctx_tokens"] > 0 and m["iter"]["gen_reqs"] > 0]
    pure = [m["iter"]["elapsed_ms"] for m in steps if m["iter"]["ctx_tokens"] == 0 and m["iter"]["gen_reqs"] > 0]
    if len(mixed) < 20 or len(pure) < 50:
        return None
    m50, p50 = q(mixed, 0.5), q(pure, 0.5)
    share = len(mixed) / (len(mixed) + len(pure))
    if not p50 or m50 < 1.5 * p50 or share < 0.1:
        return None
    budget = int(cfg_value(s, "token_budget") or 8192)
    return {
        "agent": "advisor · disaggregation",
        "title": f"Prefill slows decode: mixed steps take {m50 / p50:.1f}× as long",
        "body": f"{share:.0%} of steps carry prefill alongside decode; those take {m50:.1f} ms vs {p50:.1f} ms for "
                "decode-only steps, so every running request's next token waits behind incoming prompts.",
        "predicted": f"Smaller prefill chunks ({max(2048, budget // 2)} tokens per step) smooth inter-token latency; "
                     "if prompts are long and steady, prefill/decode disaggregation removes the interference",
        "experiment": experiment(s, "token_budget", max(2048, budget // 2), ["rag", "burst"], "e2e_p90"),
    }


def prefix_cache(s: State) -> Note | None:
    """Within each prefix family (same first ~1k prompt chars), later members should reuse the shared
    prefix. Judge per request from the engine's own cached-token count at first token."""
    rows = s.recent(s.prefill, 60)
    by_fam: dict[str, list[tuple]] = {}
    for r in rows:
        if r[0]:
            by_fam.setdefault(r[0], []).append(r)
    members = misses = 0
    shared_tokens = cached_tokens = 0
    worst = None
    for fam, rs in by_fam.items():
        if len(rs) < 3:
            continue
        core = min(r[2] for r in rs)                  # tokens every member has: a lower bound on the shared prefix
        if core < 256:
            continue
        for r in rs[1:]:                              # the first member pays for the prefix
            members += 1
            shared_tokens += core
            cached_tokens += min(r[3], core)
            if r[3] < 0.5 * core:
                misses += 1
        if worst is None or len(rs) > len(by_fam[worst]):
            worst = fam
    if members < 10:
        return None
    rate = cached_tokens / shared_tokens if shared_tokens else 1.0
    if rate >= 0.7:
        return None
    on = s.config().get("prefix_caching")
    wl = by_fam[worst][0][1] if worst else "?"
    return {
        "agent": "advisor · memory",
        "title": f"Shared prefixes are recomputed: {misses} of {members} family members missed the cache",
        "body": f"Over the last minute, requests that share a prompt prefix (e.g. the {wl} family, "
                f"{len(by_fam[worst])} requests) reused only {rate:.0%} of the shared tokens. Prefix caching is "
                f"{'on' if on else 'off'}; misses mean the prefix was evicted under memory pressure, or sessions "
                "landed on replicas that never saw it.",
        "predicted": "Prefix reuse ≥ 90%: TTFT for those requests drops by roughly the prefill time of the shared part",
        "experiment": experiment(s, "prefix_cache", None, ["agentic"], "ttft_p90") if not on else None,
    }


def expert_imbalance(s: State) -> Note | None:
    ticks = s.recent(s.moe, 30)
    if len(ticks) < 5:
        return None
    vals = [t.get("imbalance_mean") for t in ticks if t.get("imbalance_mean")]
    worst = [t.get("imbalance_max") for t in ticks if t.get("imbalance_max")]
    if not vals:
        return None
    mean_imb = sum(vals) / len(vals)
    if mean_imb < 2.0:
        return None
    cfg = s.config()
    ep = (cfg.get("ep") or 1)
    ep_on = (isinstance(ep, int) and ep > 1) or ep is True
    hot = ticks[-1].get("hot") or []
    if ep_on:
        return {
            "agent": "advisor · experts",
            "title": f"Expert load is skewed {mean_imb:.1f}× across GPUs — EP ranks wait on the busiest",
            "body": f"Per layer, the busiest expert receives {mean_imb:.1f}× the mean load (worst layer "
                    f"{max(worst):.1f}×). With experts split across GPUs, every MoE layer waits for the GPU "
                    f"holding the hot experts (now {', '.join('#' + str(h) for h in hot[:4])}).",
            "predicted": "Expert-parallel load balancing (replicate hot experts): MoE layer time toward the mean",
            "experiment": experiment(s, "eplb", None, ["burst"], "tok_s"),
        }
    return {
        "agent": "advisor · experts", "status": "info",
        "title": f"Expert routing is skewed {mean_imb:.1f}× (harmless on one GPU)",
        "body": f"Per layer, the busiest expert receives {mean_imb:.1f}× the mean load (worst layer {max(worst):.1f}×; "
                f"hottest now {', '.join('#' + str(h) for h in hot[:4])}). On a single GPU all experts share the same "
                "memory and kernels, so this costs little. It matters once experts are split across GPUs "
                "(expert parallelism): then the GPU holding the hot experts sets the pace, and load balancing "
                "(EPLB) is needed.",
        "predicted": None, "experiment": None,
    }


def spec_acceptance(s: State) -> Note | None:
    steps = [m for m in s.recent(s.steps) if m.get("spec")]
    dt = sum(m["spec"].get("draft_tokens", 0) for m in steps)
    at = sum(m["spec"].get("accepted_tokens", 0) for m in steps)
    if dt < 2000:
        return None
    rate = at / dt
    if rate >= 0.5:
        return None
    k = s.config().get("spec_tokens")
    return {
        "agent": "advisor · speculative", "title": f"Draft tokens mostly rejected ({rate:.0%} accepted)",
        "body": f"{dt:,} draft tokens proposed, {at:,} accepted over {WINDOW_S:.0f}s. Verification work is wasted.",
        "predicted": f"Fewer speculative tokens (now {k}) or turn speculation off for this traffic",
        "experiment": None,
    }


def light_load(s: State) -> Note | None:
    ticks = s.recent(s.ticks, 60)
    if len(ticks) < 500:
        return None
    running = q([t.get("running") or 0 for t in ticks], 0.9) or 0
    gpus = s.recent(s.gpu, 60)
    util = q([sum(g.get("util") or 0 for g in t.get("gpus", [])) / max(1, len(t.get("gpus", []))) for t in gpus], 0.5)
    steps = s.recent(s.steps, 10)
    if running >= 4 or util is None or util > 30 or len(steps) < 20:  # engine must be up and stepping
        return None
    return {
        "agent": "advisor · load", "status": "info", "title": "Load is too light to tune",
        "body": f"p90 running requests {running:.0f}, GPU utilization {util:.0f}%. Engine settings barely matter at "
                "this load; raise traffic before drawing conclusions.",
        "predicted": None, "experiment": None,
    }


DETECTORS: dict[str, Detector] = {
    "frontend-bound": frontend_bound,
    "queue-bound": queue_bound,
    "prefill-interference": prefill_interference,
    "prefix-cache": prefix_cache,
    "spec-acceptance": spec_acceptance,
    "expert-imbalance": expert_imbalance,
    "light-load": light_load,
}


class Advisor:
    def __init__(self, ws_url: str, post_url: str):
        self.ws_url = ws_url
        self.post_url = post_url
        self.s = State()
        self.posted: dict[str, str] = {}   # detector -> last title (to update only on change)
        self.active: set[str] = set()

    def on_event(self, e: dict[str, Any]) -> None:
        t = e.get("type")
        now = time.monotonic()
        if t == "hello":
            self.s.hello = e
        elif t == "req.arrive":
            self.s.arrive[e["rid"]] = e["t_ns"]
            self.s.workload[e["rid"]] = e.get("workload") or e["rid"]
            if e.get("prefix_family"):
                self.s.family[e["rid"]] = e["prefix_family"]
        elif t == "req.first_token":
            self.s.first[e["rid"]] = e["t_ns"]
        elif t == "req.prefill_stats":
            rid = e.get("rid")
            pt, ct = e.get("prompt_tokens"), e.get("cached_tokens")
            if isinstance(pt, int) and isinstance(ct, int):
                self.s.prefill.append((now, self.s.family.pop(rid, None), self.s.workload.get(rid, ""), pt, ct))
        elif t == "moe.tick":
            self.s.moe.append((now, e))
        elif t == "engine.step":
            self.s.steps.append((now, e))
            for f in e.get("finished") or []:
                rid = f.get("rid")
                if rid and rid in self.s.first and rid in self.s.arrive:
                    ttft = (self.s.first.pop(rid) - self.s.arrive.pop(rid)) / 1e9
                    self.s.ttft.append((now, ttft, f.get("queued_s") or 0.0, f.get("prefill_s") or 0.0,
                                        self.s.workload.pop(rid, rid)))
        elif t == "engine.tick":
            self.s.ticks.append((now, e))
        elif t == "gpu.tick":
            self.s.gpu.append((now, e))

    def post(self, note: dict[str, Any]) -> None:
        try:
            req = urllib.request.Request(self.post_url, data=json.dumps(note).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=3).read()
        except Exception as e:
            print("post failed:", e, flush=True)

    def evaluate(self) -> None:
        cfg = self.s.config()
        for name, det in DETECTORS.items():
            try:
                note = det(self.s)
            except Exception as e:  # a broken detector must not stop the others
                print(f"detector {name} failed: {e}", flush=True)
                continue
            nid = f"advice-{name}"
            if note:
                note = {"id": nid, "status": note.pop("status", "proposed"), "detector": name,
                        "engine": self.s.hello.get("engine"), "config_snapshot": {
                            k: cfg.get(k) for k in ("max_num_seqs", "max_num_batched_tokens", "block_size",
                                                    "kv_cache_dtype", "prefix_caching", "tp", "dp")}, **note}
                if self.posted.get(nid) != note["title"]:
                    self.post(note)
                    self.posted[nid] = note["title"]
                self.active.add(nid)
            elif nid in self.active:
                # Condition cleared: keep the note but mark it so the board can fade it.
                self.post({"id": nid, "status": "cleared", "detector": name,
                           "agent": "advisor", "title": self.posted.get(nid, name) + " (cleared)"})
                self.active.discard(nid)
                self.posted.pop(nid, None)

    async def run(self) -> None:
        async def evaluator() -> None:
            while True:
                await asyncio.sleep(EVAL_EVERY_S)
                self.evaluate()

        asyncio.create_task(evaluator())
        while True:
            try:
                async with ClientSession() as session, session.ws_connect(self.ws_url, max_msg_size=0) as ws:
                    print("advisor connected", flush=True)
                    async for msg in ws:
                        if msg.type != WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        for e in data if isinstance(data, list) else [data]:
                            self.on_event(e)
            except Exception as e:
                print("advisor reconnecting:", e, flush=True)
                await asyncio.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live detectors that post tuning proposals to the board")
    ap.add_argument("--collector", default="ws://127.0.0.1:18200/ws")
    ap.add_argument("--post", default="http://127.0.0.1:18200/note")
    a = ap.parse_args()
    sse.install_uvloop()
    asyncio.run(Advisor(a.collector, a.post).run())


if __name__ == "__main__":
    main()
