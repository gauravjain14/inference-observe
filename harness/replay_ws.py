"""Replay a recorded collector stream to the board, as if it were live.

Serves the same endpoints as the collector (ws://HOST:PORT/ws, /health) from a recorded
`collector.jsonl`, at the original timing (or faster). The board needs no changes: it receives
the same hello, req.*, engine.step (with batch), engine.tick, gpu.tick, kv.map, moe.tick and
advisor.note events the engine produced. Hello events gain a `replay` field so the board can say
it is a recording.

  io-replay-ws runs/<run>                    # original speed, starting at the first busy moment
  io-replay-ws runs/<run> --speed 2 --start 60 --loop
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from . import sse
from .events import InFlight

FRAME_S = 0.02  # send what is due every 20 ms, batched into one frame
MOE_BACKLOG = 90  # ~3 min of moe.tick at one per ~2 s


def t_at(f, pos: int) -> int | None:
    """t_ns of the first complete line at or after byte offset `pos`."""
    f.seek(pos)
    if pos:
        f.readline()  # skip the partial line
    for _ in range(50):
        line = f.readline()
        if not line:
            return None
        try:
            t = json.loads(line).get("t_ns")
        except json.JSONDecodeError:
            continue
        if t:
            return t
    return None


def seek_time(f, size: int, target_ns: int) -> int:
    """Byte offset of the first line with t_ns >= target_ns (binary search; t_ns ~ monotonic)."""
    lo, hi = 0, size
    while hi - lo > 1 << 16:
        mid = (lo + hi) // 2
        t = t_at(f, mid)
        if t is None or t >= target_ns:
            hi = mid
        else:
            lo = mid
    f.seek(lo)
    if lo:
        f.readline()
    return f.tell()


class Replayer:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.path = a.run / "collector.jsonl" if a.run.is_dir() else a.run
        self.subs: set[asyncio.Queue] = set()
        self.hello: dict[str, Any] | None = None
        self.last: dict[str, dict[str, Any]] = {}   # last engine.tick / gpu.tick / kv.map / moe.tick
        self.moe_hist: deque = deque(maxlen=MOE_BACKLOG)  # recent moe.tick, so the expert heatmap starts full
        self.inflight = InFlight()
        self.notes: dict[str, dict[str, Any]] = {}
        if a.notes and a.notes.exists():
            for line in a.notes.read_text().splitlines():
                try:
                    ev = json.loads(line)
                    self.notes[ev["id"]] = ev
                except (json.JSONDecodeError, KeyError):
                    pass
        self.position_ns = 0
        self.rec_start_ns = 0

    def mark(self, ev: dict[str, Any]) -> dict[str, Any]:
        if ev.get("type") == "hello":
            rec = time.strftime("%Y-%m-%d %H:%M", time.gmtime((ev.get("t_ns") or 0) / 1e9))
            ev = {**ev, "label": self.a.label or ev.get("label") or "Recorded specimen",
                  "replay": {"file": self.path.name, "speed": self.a.speed, "recorded_utc": rec}}
            self.hello = ev
        elif ev.get("type") in ("engine.tick", "gpu.tick", "kv.map", "moe.tick"):
            self.last[ev["type"]] = ev
            if ev["type"] == "moe.tick":
                self.moe_hist.append(ev)
        elif ev.get("type") == "advisor.note" and ev.get("id"):
            self.notes[ev["id"]] = ev
        self.inflight.add(ev)
        return ev

    def broadcast(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        frame = json.dumps(batch, separators=(",", ":"), default=str)
        for q in list(self.subs):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass  # a slow tab misses frames rather than stalling the replay

    def first_busy_ns(self, f, size: int) -> int:
        """Skip engine startup: first engine.step after the first hello."""
        f.seek(0)
        for line in f:
            if '"type":"engine.step"' in line:
                try:
                    return json.loads(line)["t_ns"]
                except (json.JSONDecodeError, KeyError):
                    continue
        return 0

    async def run(self) -> None:
        size = self.path.stat().st_size
        while True:
            with open(self.path, "r") as f:
                # Prime hello and the latest ticks from the start of the file.
                for line in f:
                    ev = json.loads(line)
                    if ev.get("type") == "hello":
                        self.mark(ev)
                    if self.hello and ev.get("type") == "engine.step":
                        break
                start_ns = self.first_busy_ns(f, size) + int(self.a.start * 1e9)
                # Pre-load ~3 min before the start point: expert history, and the requests still in flight.
                f.seek(seek_time(f, size, start_ns - int(MOE_BACKLOG * 2.2e9)))
                for line in f:
                    if '"type":"moe.tick"' in line or '"type":"req.' in line:
                        ev = json.loads(line)
                        if ev.get("t_ns", 0) >= start_ns:
                            break
                        if ev["type"] == "moe.tick":
                            self.mark(ev)
                        else:
                            self.inflight.add(ev)
                    elif '"t_ns":' in line and json.loads(line).get("t_ns", 0) >= start_ns:
                        break
                f.seek(seek_time(f, size, start_ns))
                self.rec_start_ns = start_ns
                wall0 = time.monotonic()
                batch: list[dict[str, Any]] = []
                next_flush = wall0 + FRAME_S
                for line in f:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = ev.get("t_ns")
                    if t:
                        due = wall0 + (t - start_ns) / 1e9 / self.a.speed
                        now = time.monotonic()
                        if due > now:
                            self.broadcast(batch)
                            batch = []
                            await asyncio.sleep(due - now)
                            next_flush = time.monotonic() + FRAME_S
                        self.position_ns = t
                    batch.append(self.mark(ev))
                    if time.monotonic() >= next_flush or len(batch) >= 500:
                        self.broadcast(batch)
                        batch = []
                        next_flush = time.monotonic() + FRAME_S
                        await asyncio.sleep(0)
                self.broadcast(batch)
            if not self.a.loop:
                print("replay finished", flush=True)
                return
            print("replay looping", flush=True)

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
        await ws.prepare(request)
        backlog = ([e for e in (self.hello, *(v for k, v in self.last.items() if k != "moe.tick")) if e]
                   + list(self.moe_hist) + list(self.notes.values()) + self.inflight.backlog())
        if backlog:
            await ws.send_str(json.dumps(backlog, default=str))
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self.subs.add(q)

        async def pump() -> None:
            while True:
                await ws.send_str(await q.get())

        sender = asyncio.create_task(pump())
        try:
            async for msg in ws:
                if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            sender.cancel()
            self.subs.discard(q)
        return ws

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "hello": self.hello, "last_tick": self.last.get("engine.tick"),
            "replay": {"file": self.path.name, "speed": self.a.speed,
                       "position_s": round((self.position_ns - self.rec_start_ns) / 1e9, 1) if self.position_ns else 0},
        }, dumps=lambda o: json.dumps(o, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a recorded collector stream to the board")
    ap.add_argument("run", type=Path, help="run directory (with collector.jsonl) or the jsonl file")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0, help="seconds after the first engine step")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--label", default=None, help="caption on the board's plate (default: the recording's own)")
    ap.add_argument("--notes", type=Path, default=None, help="notes.jsonl for the ledger and Field notes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18200)
    a = ap.parse_args()
    sse.install_uvloop()
    r = Replayer(a)
    app = web.Application()
    app.router.add_get("/ws", r.ws)
    app.router.add_get("/health", r.health)

    async def start(app: web.Application) -> None:
        asyncio.create_task(r.run())

    app.on_startup.append(start)
    print(f"replaying {r.path} on ws://{a.host}:{a.port}/ws at {a.speed}x", flush=True)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
