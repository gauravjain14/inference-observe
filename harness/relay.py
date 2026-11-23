"""Read-only public relay for the board stream.

The relay holds one connection to the collector and sends the same frames to every viewer. Viewers can
only watch:
- it serves GET /ws and nothing else;
- it disconnects any viewer that sends data;
- it caps viewers overall and per address;
- it disconnects viewers that fall behind, so a slow or stalled client never grows memory;
- it accepts browsers only from the listed origins (--allow-origin);
- it removes local file paths from the stream.

  io-relay --upstream ws://127.0.0.1:18220/ws --port 18240 \\
      --allow-origin https://<user>.github.io --trust-cloudflare
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Any

from aiohttp import ClientSession, WSCloseCode, WSMsgType, web

from .events import InFlight

QUEUE_FRAMES = 64  # frames a viewer may fall behind before it is disconnected


class Relay:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.viewers: dict[web.WebSocketResponse, tuple[asyncio.Queue[str], web.Request]] = {}
        self.per_ip: dict[str, int] = defaultdict(int)
        # what a viewer joining now needs: the latest of each state event, recent expert routing, notes and
        # the lifecycle events of requests still in flight
        self.hello: dict[str, Any] | None = None
        self.latest: dict[str, dict[str, Any]] = {}
        self.moe: deque = deque(maxlen=90)
        self.notes: dict[str, dict[str, Any]] = {}
        self.inflight = InFlight()
        self.stats = {"frames": 0, "bytes": 0, "rejected": 0, "dropped_slow": 0, "closed_for_input": 0}

    # ---- upstream ---------------------------------------------------------------------------
    @staticmethod
    def clean(ev: dict[str, Any]) -> dict[str, Any]:
        if ev.get("type") == "hello" and isinstance(ev.get("replay"), dict) and ev["replay"].get("file"):
            ev = {**ev, "replay": {**ev["replay"], "file": str(ev["replay"]["file"]).rsplit("/", 1)[-1]}}
        return ev

    def remember(self, ev: dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "hello":
            self.hello = ev
        elif t in ("engine.tick", "gpu.tick"):
            self.latest[t] = ev
        elif t == "kv.map":
            self.latest[f"kv.map:{ev.get('instance', 'engine')}"] = ev
        elif t == "moe.tick":
            self.moe.append(ev)
        elif t == "advisor.note" and ev.get("id"):
            self.notes[ev["id"]] = ev
        self.inflight.add(ev)

    async def upstream(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with ClientSession() as s, s.ws_connect(self.a.upstream, heartbeat=20, max_msg_size=0) as ws:
                    backoff = 1.0
                    async for msg in ws:
                        if msg.type != WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        evs = data if isinstance(data, list) else [data]
                        evs = [self.clean(e) for e in evs if isinstance(e, dict)]
                        for e in evs:
                            self.remember(e)
                        self.fanout(json.dumps(evs if isinstance(data, list) else evs[0], separators=(",", ":")))
            except Exception:
                pass
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def fanout(self, frame: str) -> None:
        self.stats["frames"] += 1
        self.stats["bytes"] += len(frame)
        for ws, (q, request) in list(self.viewers.items()):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:  # fell behind: drop the connection at once (a polite close would wait on it)
                self.stats["dropped_slow"] += 1
                self.viewers.pop(ws, None)
                if request.transport is not None:
                    request.transport.abort()

    def snapshot(self) -> list[str]:
        out = []
        if self.hello:
            out.append(json.dumps(self.hello, separators=(",", ":")))
        rest = list(self.latest.values()) + list(self.moe) + list(self.notes.values()) + self.inflight.backlog()
        if rest:
            out.append(json.dumps(rest, separators=(",", ":")))
        return out

    # ---- viewers ----------------------------------------------------------------------------
    def client_ip(self, request: web.Request) -> str:
        if self.a.trust_cloudflare and request.headers.get("CF-Connecting-IP"):
            return request.headers["CF-Connecting-IP"].strip()
        return request.remote or "?"

    async def ws(self, request: web.Request) -> web.StreamResponse:
        origin = request.headers.get("Origin")
        if self.a.allow_origin and origin not in self.a.allow_origin:
            self.stats["rejected"] += 1
            return web.Response(status=403, text="origin not allowed")
        ip = self.client_ip(request)
        if len(self.viewers) >= self.a.max_viewers or self.per_ip[ip] >= self.a.max_per_ip:
            self.stats["rejected"] += 1
            return web.Response(status=429, text="too many viewers")
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=1024, compress=True)
        await ws.prepare(request)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_FRAMES)
        self.per_ip[ip] += 1
        for frame in self.snapshot():
            q.put_nowait(frame)
        self.viewers[ws] = (q, request)

        async def send() -> None:
            while not ws.closed:
                frame = await q.get()
                await ws.send_str(frame)

        sender = asyncio.create_task(send())
        try:
            async for msg in ws:  # viewers only watch: any data from them ends the connection
                if msg.type in (WSMsgType.TEXT, WSMsgType.BINARY):
                    self.stats["closed_for_input"] += 1
                    try:
                        await asyncio.wait_for(ws.close(code=WSCloseCode.POLICY_VIOLATION, message=b"read-only"), 2)
                    except asyncio.TimeoutError:
                        if request.transport is not None:
                            request.transport.abort()
                break
        finally:
            sender.cancel()
            self.viewers.pop(ws, None)
            self.per_ip[ip] -= 1
            if self.per_ip[ip] <= 0:
                self.per_ip.pop(ip, None)
        return ws

    async def report(self) -> None:
        while True:
            await asyncio.sleep(60)
            print(json.dumps({"t": time.strftime("%H:%M:%S"), "viewers": len(self.viewers),
                              "addresses": len(self.per_ip), **self.stats}), flush=True)


def build_app(a: argparse.Namespace) -> web.Application:
    relay = Relay(a)
    app = web.Application(client_max_size=1024)
    app.router.add_get("/ws", relay.ws)

    async def start(app: web.Application) -> None:
        app["tasks"] = [asyncio.create_task(relay.upstream()), asyncio.create_task(relay.report())]

    async def stop(app: web.Application) -> None:
        for t in app["tasks"]:
            t.cancel()

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app["relay"] = relay
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only public relay for the board stream")
    ap.add_argument("--upstream", default="ws://127.0.0.1:18220/ws", help="the collector's board stream")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18240)
    ap.add_argument("--allow-origin", action="append", default=[],
                    help="page origin allowed to connect, e.g. https://<user>.github.io (repeatable; "
                         "without it any origin may connect)")
    ap.add_argument("--max-viewers", type=int, default=200)
    ap.add_argument("--max-per-ip", type=int, default=4)
    ap.add_argument("--trust-cloudflare", action="store_true",
                    help="take the viewer's address from CF-Connecting-IP (only behind a Cloudflare tunnel)")
    a = ap.parse_args()
    web.run_app(build_app(a), host=a.host, port=a.port, print=None, shutdown_timeout=2.0)


if __name__ == "__main__":
    main()
