"""Flow events: one JSON object per line, written to disk and fanned out live.

Version 0 is the proxy's view only (what crosses the wire). Engine-side events
(scheduler steps, KV blocks, GPU samples) join in Phase 2 under the same envelope:
    {"v": 0, "type": "...", "t_ns": <wall clock ns>, "src": "proxy", ...fields}
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable

FLOW_VERSION = 0


def now_ns() -> int:
    return time.time_ns()


# Lifecycle events kept for requests still in flight, so a board that connects mid-run
# can place them correctly (traffic class, prompt size, phase) instead of guessing.
INFLIGHT_TYPES = ("req.arrive", "req.admit", "req.route", "req.prefill_start", "req.prefill_done",
                  "req.prefill_stats", "req.decode_start", "req.first_token")
INFLIGHT_END = ("req.done", "req.cancel", "req.error")


class InFlight:
    def __init__(self, cap: int = 5000):
        self.cap = cap
        self.events: dict[str, list[dict[str, Any]]] = {}

    def add(self, ev: dict[str, Any]) -> None:
        rid, t = ev.get("rid"), ev.get("type")
        if not rid:
            return
        if t in INFLIGHT_END:
            self.events.pop(rid, None)
        elif t in INFLIGHT_TYPES:
            self.events.setdefault(rid, []).append(ev)
            if len(self.events) > self.cap:
                self.events.pop(next(iter(self.events)))

    def backlog(self) -> list[dict[str, Any]]:
        return sorted((e for evs in self.events.values() for e in evs), key=lambda e: e.get("t_ns", 0))


class EventSink:
    """Appends events to a JSONL file and pushes them to live subscribers.

    Subscribers get bounded queues; a slow subscriber loses events rather than
    slowing the request path.
    """

    def __init__(self, path: Path | None, src: str):
        self.src = src
        self._fh = open(path, "a", buffering=1) if path else None
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self.dropped = 0
        self.on_publish: Callable[[dict[str, Any]], None] | None = None

    def emit(self, type_: str, **fields: Any) -> dict[str, Any]:
        ev = {"v": FLOW_VERSION, "type": type_, "t_ns": now_ns(), "src": self.src, **fields}
        return self.publish(ev)

    def publish(self, ev: dict[str, Any]) -> dict[str, Any]:
        """Write and fan out an already-built event (keeps its own src and t_ns)."""
        if self.on_publish:
            self.on_publish(ev)
        if self._fh:
            self._fh.write(json.dumps(ev, separators=(",", ":"), default=str) + "\n")
        for q in self._subs:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                self.dropped += 1
        return ev

    def subscribe(self, maxsize: int = 10_000) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(q)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
