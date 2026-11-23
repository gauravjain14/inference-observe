"""Collector: merges every signal source into one live stream for the board.

Sources
  proxy    per-request lifecycle events, from the proxy's /_events WebSocket
  tap      per-step engine data, UDP datagrams from an in-process tap (io_taps)
  metrics  the engine's Prometheus endpoint, polled; used when no tap is talking
  gpu      NVML samples for the GPUs the engine runs on

Output
  ws://HOST:PORT/ws   hello, then every event (req.*, engine.step, engine.tick, gpu.tick, ...)
  <run>/collector.jsonl  the same stream, for exact replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

from . import sse
from .events import FLOW_VERSION, EventSink, InFlight, now_ns

TICK_HZ = 10
RATE_WINDOW_S = 2.0
HIT_WINDOW_S = 60.0  # prefix-cache hit rate: prompts that started prefill in the last minute
TAP_FRESH_S = 2.0

# Prometheus gauges used when no tap is reporting. Values are summed across label sets
# (e.g. DP ranks), except kv usage which is averaged.
PROM = {
    "vllm": {"running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting",
             "kv_usage": "vllm:kv_cache_usage_perc", "preemptions_total": "vllm:num_preemptions_total"},
    "sglang": {"running": "sglang:num_running_reqs", "waiting": "sglang:num_queue_reqs",
               "kv_usage": "sglang:token_usage", "gen_tok_s": "sglang:gen_throughput",
               "prefix_hit_rate": "sglang:cache_hit_rate", "spec_accept_len": "sglang:spec_accept_length"},
}
AVERAGED = {"kv_usage", "prefix_hit_rate", "spec_accept_len"}
PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


def parse_prometheus(text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = PROM_LINE.match(line)
        if m:
            try:
                out.setdefault(m.group(1), []).append(float(m.group(3)))
            except ValueError:
                pass
    return out


class TapProtocol(asyncio.DatagramProtocol):
    def __init__(self, collector: "Collector"):
        self.c = collector

    def datagram_received(self, data: bytes, addr: Any) -> None:
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return
        self.c.on_tap(msg)


class Collector:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sink = EventSink(args.out / "collector.jsonl" if args.out else None, src="collector")
        self.gpu_ids = [int(x) for x in args.gpus.split(",")] if args.gpus else []
        self.hello: dict[str, Any] = {
            "v": FLOW_VERSION, "type": "hello", "src": "collector",
            "engine": args.engine, "engine_version": None, "model": args.model,
            "gpus": len(self.gpu_ids), "gpu_ids": self.gpu_ids,
            "coverage": {"requests": False, "steps": False, "kv": False, "spec": False, "gpu": False,
                         "moe": False},
            "config": {},
            "instances": {},
        }
        if args.label:
            self.hello["label"] = args.label
        self.inst_last: dict[str, dict[str, Any]] = {}
        self.inst_t: dict[str, float] = {}
        self.steps: deque[tuple[float, dict[str, Any]]] = deque()
        self.last_step: dict[str, Any] | None = None
        self.last_tap_t = 0.0
        self.preemptions_total = 0
        self.prom: dict[str, float | None] = {}
        self.active: set[str] = set()
        self.last_tick: dict[str, Any] | None = None
        self.last_gpu: dict[str, Any] | None = None
        self.notes: dict[str, dict[str, Any]] = {}
        self.moe_hist: deque = deque(maxlen=90)  # recent moe.tick, sent on connect for the expert heatmap
        self.inflight = InFlight()

    # ---- publishing -------------------------------------------------------------
    def publish(self, ev: dict[str, Any]) -> None:
        self.inflight.add(ev)
        self.sink.publish(ev)

    def set_coverage(self, key: str, on: bool = True) -> None:
        if self.hello["coverage"].get(key) != on:
            self.hello["coverage"][key] = on
            self.publish({**self.hello, "t_ns": now_ns()})

    # ---- proxy ------------------------------------------------------------------
    async def proxy_loop(self) -> None:
        url = self.args.proxy.rstrip("/") + "/_events"
        while True:
            try:
                async with ClientSession() as s, s.ws_connect(url, heartbeat=20) as ws:
                    self.set_coverage("requests")
                    async for msg in ws:
                        if msg.type != WSMsgType.TEXT:
                            continue
                        ev = json.loads(msg.data)
                        t = ev.get("type", "")
                        rid = ev.get("rid")
                        if t == "proxy.hello":
                            self.close_lost(set(ev.get("inflight") or []))
                            continue
                        if t == "req.arrive" and rid:
                            self.active.add(rid)
                        elif t in ("req.done", "req.cancel", "req.error") and rid:
                            self.active.discard(rid)
                        self.publish(ev)
            except (OSError, asyncio.TimeoutError, Exception):
                self.set_coverage("requests", False)
                await asyncio.sleep(1.0)

    def close_lost(self, serving: set[str]) -> None:
        """End requests the proxy is no longer serving (it restarted, or its event stream dropped them)."""
        for rid in sorted(self.active - serving):
            self.active.discard(rid)
            self.publish({"v": FLOW_VERSION, "type": "req.error", "t_ns": now_ns(), "src": "collector", "rid": rid,
                          "stage": "lost", "status": None,
                          "detail": "the proxy stopped serving this request without an end event; outcome unknown"})

    # ---- tap --------------------------------------------------------------------
    def match_rid(self, engine_request_id: str) -> str | None:
        for rid in self.active:
            if engine_request_id.endswith(rid) or f"-{rid}-" in engine_request_id or engine_request_id.startswith(rid):
                return rid
        m = re.match(r"^(?:chatcmpl|cmpl)-(.+)$", engine_request_id)
        return m.group(1) if m else None

    def on_tap(self, msg: dict[str, Any]) -> None:
        t = msg.get("type")
        if t in ("tap.hello", "tap.ready"):
            before = json.dumps(self.hello, sort_keys=True, default=str)
            self.hello["engine_version"] = msg.get("engine_version") or self.hello["engine_version"]
            self.hello["config"].update({k: v for k, v in (msg.get("config") or {}).items() if v is not None})
            if not self.hello["model"]:
                self.hello["model"] = self.hello["config"].get("model")
            inst = msg.get("instance", "engine")
            self.hello["instances"][inst] = {"role": inst.split("-")[0], "config": msg.get("config") or {},
                                             "url": msg.get("url")}
            if json.dumps(self.hello, sort_keys=True, default=str) != before:  # taps repeat their hello
                self.publish({**self.hello, "t_ns": now_ns()})
            return
        if t != "engine.step":
            if t == "moe.tick":
                self.set_coverage("moe")
                self.moe_hist.append(msg)
            self.publish(msg)
            return
        now = time.monotonic()
        self.last_tap_t = now
        if not self.hello["coverage"]["steps"]:
            self.set_coverage("steps")
            self.set_coverage("kv")
        if msg.get("spec") and not self.hello["coverage"]["spec"]:
            self.set_coverage("spec")
        self.preemptions_total += msg.get("preempted") or 0
        for f in msg.get("finished") or []:
            rid = self.match_rid(str(f.get("request_id", "")))
            if rid:
                f["rid"] = rid
        self.steps.append((now, msg))
        self.last_step = msg
        inst = msg.get("instance", "engine")
        self.inst_last[inst] = msg
        self.inst_t[inst] = now
        self.publish(msg)

    # ---- metrics fallback ---------------------------------------------------------
    async def metrics_loop(self) -> None:
        names = PROM.get(self.args.engine, {})
        url = self.args.engine_url.rstrip("/") + "/metrics"
        async with ClientSession(timeout=ClientTimeout(total=2)) as s:
            while True:
                try:
                    async with s.get(url) as r:
                        vals = parse_prometheus(await r.text())
                    for key, name in names.items():
                        xs = vals.get(name)
                        self.prom[key] = None if not xs else (sum(xs) / len(xs) if key in AVERAGED else sum(xs))
                    if self.prom.get("kv_usage") is not None:
                        self.set_coverage("kv")
                except Exception:
                    pass
                await asyncio.sleep(0.5)

    # ---- gpu ----------------------------------------------------------------------
    async def gpu_loop(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = [(i, pynvml.nvmlDeviceGetHandleByIndex(i)) for i in self.gpu_ids]
        except Exception as e:
            self.publish({"v": FLOW_VERSION, "type": "collector.warn", "t_ns": now_ns(), "src": "collector",
                          "detail": f"NVML unavailable: {e}"})
            return
        self.set_coverage("gpu", bool(handles))
        while True:
            gpus = []
            for i, h in handles:
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    gpus.append({
                        "i": i, "util": u.gpu, "mem_util": u.memory,
                        "mem_used_gb": round(m.used / 2**30, 2), "mem_total_gb": round(m.total / 2**30, 2),
                        "power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1),
                        "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                    })
                except Exception:
                    gpus.append({"i": i})
            self.last_gpu = {"v": FLOW_VERSION, "type": "gpu.tick", "t_ns": now_ns(), "src": "collector", "gpus": gpus}
            self.publish(self.last_gpu)
            await asyncio.sleep(1 / TICK_HZ)

    # ---- engine tick ----------------------------------------------------------------
    def window(self, seconds: float) -> list[dict[str, Any]]:
        now = time.monotonic()
        while self.steps and now - self.steps[0][0] > HIT_WINDOW_S:
            self.steps.popleft()
        return [m for t, m in self.steps if now - t <= seconds]

    async def tick_loop(self) -> None:
        while True:
            await asyncio.sleep(1 / TICK_HZ)
            tap_live = time.monotonic() - self.last_tap_t < TAP_FRESH_S
            tick: dict[str, Any] = {"v": FLOW_VERSION, "type": "engine.tick", "t_ns": now_ns(), "src": "collector",
                                    "engine": self.args.engine, "source": "tap" if tap_live else "metrics"}
            if tap_live and self.last_step:
                recent = self.window(RATE_WINDOW_S)
                hits = self.window(HIT_WINDOW_S)
                now = time.monotonic()
                fresh = {k: m for k, m in self.inst_last.items() if now - self.inst_t.get(k, 0) < 5.0}
                kvs = [m.get("kv_usage") for m in self.inst_last.values() if m.get("kv_usage") is not None]
                s = {"running": sum(m.get("running") or 0 for m in fresh.values()),
                     "waiting": sum(m.get("waiting") or 0 for m in fresh.values()),
                     "kv_usage": sum(kvs) / len(kvs) if kvs else None}
                instances = {}
                for name, last in self.inst_last.items():
                    mine = [m for m in recent if m.get("instance", "engine") == name]
                    live = name in fresh
                    instances[name] = {
                        "role": name.split("-")[0],
                        "running": last.get("running") if live else 0,
                        "waiting": last.get("waiting") if live else 0,
                        "kv_usage": last.get("kv_usage"),
                        "steps_per_s": round(len(mine) / RATE_WINDOW_S, 1),
                        "gen_tok_s": round(sum(m.get("gen_tokens") or 0 for m in mine) / RATE_WINDOW_S, 1),
                        "prompt_tok_s": round(sum((m.get("iter") or {}).get("ctx_tokens") or 0 for m in mine) / RATE_WINDOW_S, 1),
                    }
                tick["instances"] = instances
                gen = sum(m.get("gen_tokens") or 0 for m in recent)
                ctx = sum((m.get("iter") or {}).get("ctx_tokens") or 0 for m in recent)
                q = sum((m.get("prefix") or {}).get("queries", 0) for m in hits)
                h = sum((m.get("prefix") or {}).get("hits", 0) for m in hits)
                dt = sum((m.get("spec") or {}).get("draft_tokens", 0) for m in hits)
                at = sum((m.get("spec") or {}).get("accepted_tokens", 0) for m in hits)
                dr = sum((m.get("spec") or {}).get("drafts", 0) for m in hits)
                tick.update(
                    running=s.get("running"), waiting=s.get("waiting"), kv_usage=s.get("kv_usage"),
                    prefix_hit_rate=(h / q) if q else None,
                    prompt_tok_s=round(ctx / RATE_WINDOW_S, 1), gen_tok_s=round(gen / RATE_WINDOW_S, 1),
                    steps_per_s=round(len(recent) / RATE_WINDOW_S, 1),
                    preemptions_total=self.preemptions_total,
                    spec={"accept_rate": at / dt, "mean_accept_len": 1 + at / dr} if dt and dr else None,
                )
            else:
                p = self.prom
                tick.update(
                    running=p.get("running"), waiting=p.get("waiting"), kv_usage=p.get("kv_usage"),
                    prefix_hit_rate=p.get("prefix_hit_rate"), prompt_tok_s=None, gen_tok_s=p.get("gen_tok_s"),
                    steps_per_s=None, preemptions_total=p.get("preemptions_total"),
                    spec={"accept_rate": None, "mean_accept_len": p["spec_accept_len"]} if p.get("spec_accept_len") else None,
                )
            tick["active_requests"] = len(self.active)
            self.last_tick = tick
            self.publish(tick)

    # ---- board websocket ----------------------------------------------------------------
    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
        await ws.prepare(request)
        await ws.send_str(json.dumps({**self.hello, "t_ns": now_ns()}, default=str))
        backlog = ([ev for ev in (self.last_tick, self.last_gpu) if ev] + list(self.moe_hist) + list(self.notes.values())
                   + self.inflight.backlog())
        if backlog:
            await ws.send_str(json.dumps(backlog, default=str))
        q = self.sink.subscribe(maxsize=50_000)

        async def pump() -> None:
            while True:
                batch = [await q.get()]
                while not q.empty() and len(batch) < 500:
                    batch.append(q.get_nowait())
                # One frame per batch keeps per-message overhead low at thousands of events/s.
                await ws.send_str(json.dumps(batch, separators=(",", ":"), default=str))

        sender = asyncio.create_task(pump())
        try:
            async for msg in ws:
                if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            sender.cancel()
            self.sink.unsubscribe(q)
        return ws

    async def note(self, request: web.Request) -> web.Response:
        """Advisor/experiment notes, shown in the board's Field notes panel.

        Body: {id?, agent, title, body, predicted?, measured?, region?, status?}
        Re-posting the same id updates that note (e.g. when a measurement lands).
        """
        body = await request.json()
        ev = {"v": FLOW_VERSION, "type": "advisor.note", "t_ns": now_ns(), "src": body.get("agent", "advisor"),
              "id": body.get("id") or f"note-{now_ns()}", **{k: v for k, v in body.items() if k != "id"}}
        self.notes[ev["id"]] = ev
        if self.args.notes:
            with open(self.args.notes, "a") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        self.publish(ev)
        return web.json_response({"ok": True, "id": ev["id"]})

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"hello": self.hello, "last_tick": self.last_tick, "dropped": self.sink.dropped,
                                  "active": len(self.active)}, dumps=lambda o: json.dumps(o, default=str))

    async def start(self, app: web.Application) -> None:
        if self.args.notes and self.args.notes.exists():
            for line in self.args.notes.read_text().splitlines():
                try:
                    ev = json.loads(line)
                    self.notes[ev["id"]] = ev
                except (json.JSONDecodeError, KeyError):
                    pass
        loop = asyncio.get_running_loop()
        for attempt in range(30):  # a collector that is still shutting down may hold the port briefly
            try:
                await loop.create_datagram_endpoint(lambda: TapProtocol(self), local_addr=("127.0.0.1", self.args.tap_port))
                break
            except OSError:
                if attempt == 29:
                    raise
                await asyncio.sleep(0.5)
        self.publish({**self.hello, "t_ns": now_ns()})
        for coro in (self.proxy_loop(), self.metrics_loop(), self.gpu_loop(), self.tick_loop()):
            asyncio.create_task(coro)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge proxy, engine tap, metrics and GPU signals into one stream")
    ap.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    ap.add_argument("--engine-url", default="http://127.0.0.1:18100")
    ap.add_argument("--proxy", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU indices the engine uses")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--tap-port", type=int, default=18201)
    ap.add_argument("--out", type=Path, help="run directory for collector.jsonl")
    ap.add_argument("--label", help="caption on the board's plate, e.g. the traffic source")
    ap.add_argument("--notes", type=Path, help="notes log shared across runs (loaded at start, appended on post)")
    args = ap.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    sse.install_uvloop()
    c = Collector(args)
    app = web.Application()
    app.on_startup.append(c.start)
    app.router.add_get("/ws", c.ws_handler)
    app.router.add_get("/health", c.health)
    app.router.add_post("/note", c.note)
    web.run_app(app, host=args.host, port=args.port, print=None, shutdown_timeout=2.0)  # boards reconnect


if __name__ == "__main__":
    main()
