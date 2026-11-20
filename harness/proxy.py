"""OpenAI-compatible proxy that sits in front of any engine.

It does three jobs:
  1. Timestamps every request and every streamed chunk, so the board can move a
     request while it is still in flight (engine traces only arrive at the end).
  2. Holds an admission cap that can be changed live (POST /_control), so an
     advisor's "apply" can take effect without restarting the engine.
  3. Publishes those events to a JSONL file and to live WebSocket subscribers
     (GET /_events).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, TCPConnector, WSMsgType, web

from . import sse
from .events import EventSink, now_ns

CHUNK_FLUSH_NS = 100_000_000  # coalesce token events per request to ~10 Hz

STANDARD_CHUNK_KEYS = {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}


class Gate:
    """Admission cap that can be resized while running. limit=None means no cap.

    Waiters are served first-come first-served. release() and set_limit() are
    synchronous so they are safe to call from finally blocks during cancellation.
    """

    def __init__(self, limit: int | None = None):
        self.limit = limit
        self.inflight = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def waiting(self) -> int:
        return sum(1 for f in self._waiters if not f.done())

    def _has_room(self) -> bool:
        return self.limit is None or self.inflight < self.limit

    async def acquire(self) -> None:
        if self._has_room() and not self._waiters:
            self.inflight += 1
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # a slot was granted just as we were cancelled; hand it on
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        self.inflight -= 1
        self._wake()

    def set_limit(self, limit: int | None) -> None:
        self.limit = limit
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self._has_room():
            fut = self._waiters.popleft()
            if not fut.done():
                self.inflight += 1
                fut.set_result(None)


def _prompt_shape(body: dict[str, Any]) -> dict[str, Any]:
    msgs = body.get("messages")
    if isinstance(msgs, list):
        chars = 0
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str):
                chars += len(c)
            elif isinstance(c, list):
                chars += sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
        return {"n_messages": len(msgs), "prompt_chars": chars}
    p = body.get("prompt")
    return {"n_messages": 0, "prompt_chars": len(p) if isinstance(p, str) else None}


def _prefix_family(body: dict[str, Any], n_chars: int = 1024) -> dict[str, Any]:
    """Fingerprint of the prompt's first ~1k characters. Requests in one family should share
    prefix-cached KV; the engine's reported cached tokens then say whether they actually did."""
    msgs = body.get("messages")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str):
                parts.append(c)
            if sum(len(x) for x in parts) >= n_chars:
                break
        text = "\x1e".join(parts)
    else:
        text = body.get("prompt") if isinstance(body.get("prompt"), str) else ""
    if not text:
        return {}
    head = text[:n_chars].encode(errors="ignore")
    return {"prefix_family": hashlib.blake2b(head, digest_size=5).hexdigest(), "prefix_chars": len(head)}


def _chunk_text_chars(obj: dict[str, Any]) -> tuple[int, str | None]:
    n = 0
    finish = None
    for c in obj.get("choices") or []:
        delta = c.get("delta") or {}
        for key in ("content", "reasoning_content", "reasoning"):
            v = delta.get(key)
            if isinstance(v, str):
                n += len(v)
        t = c.get("text")
        if isinstance(t, str):
            n += len(t)
        finish = c.get("finish_reason") or finish
    return n, finish


class Proxy:
    def __init__(self, upstream: str, sink: EventSink, max_inflight: int | None,
                 prefill: list[str] | None = None, decode: list[str] | None = None):
        self.upstream = upstream.rstrip("/")
        self.sink = sink
        self.gate = Gate(max_inflight)
        self.session: ClientSession | None = None
        # Prefill/decode disaggregation (vLLM KV-connector protocol): when both lists are set,
        # each request is prefilled on one instance and decoded on another.
        self.prefill = [u.rstrip("/") for u in prefill or []]
        self.decode = [u.rstrip("/") for u in decode or []]
        self._rr = {"p": 0, "d": 0}
        self.rid_in_body = False
        # Independent replicas: route each request to the replica with the fewest in flight.
        self.replicas: list[str] = []
        self.replica_inflight: dict[str, int] = {}

    def set_replicas(self, urls: list[str]) -> None:
        self.replicas = [u.rstrip("/") for u in urls]
        self.replica_inflight = {u: 0 for u in self.replicas}

    def _pick(self, pool: list[str], key: str) -> str:
        url = pool[self._rr[key] % len(pool)]
        self._rr[key] += 1
        return url

    async def _remote_prefill(self, rid: str, path: str, body: dict, headers: dict) -> tuple[dict | None, web.Response | None]:
        """Run the prefill half on a prefill instance; returns kv_transfer_params for the decode half."""
        assert self.session
        p_url = self._pick(self.prefill, "p")
        pbody = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        pbody["max_tokens"] = 1
        if "max_completion_tokens" in pbody:
            pbody["max_completion_tokens"] = 1
        pbody["stream"] = False
        pbody["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False,
                                       "remote_engine_id": None, "remote_block_ids": None,
                                       "remote_host": None, "remote_port": None}
        t0 = now_ns()
        self.sink.emit("req.prefill_start", rid=rid, worker=p_url)
        async with self.session.post(p_url + path, json=pbody, headers=headers) as r:
            raw = await r.read()
            if r.status != 200:
                self.sink.emit("req.error", rid=rid, status=r.status, stage="prefill", detail=raw[:500].decode(errors="replace"))
                return None, web.Response(status=r.status, body=raw, content_type=r.content_type)
        obj = json.loads(raw)
        ktp = obj.get("kv_transfer_params") or {}
        usage = obj.get("usage") or {}
        self.sink.emit("req.prefill_done", rid=rid, worker=p_url, prefill_ms=(now_ns() - t0) / 1e6,
                       prompt_tokens=usage.get("prompt_tokens"),
                       cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                       kv_blocks=len(ktp.get("remote_block_ids") or []) if isinstance(ktp.get("remote_block_ids"), list) else None)
        return ktp, None

    async def start(self, app: web.Application) -> None:
        # limit=0: the default pool (100) would cap upstream concurrency below the engine's own limit.
        # force_close: a fresh localhost connection per request avoids the keep-alive race where the
        # engine closes an idle connection just as it is reused ("Server disconnected" -> 502).
        self.session = ClientSession(connector=TCPConnector(limit=0, force_close=True),
                                     timeout=ClientTimeout(total=None, sock_connect=10, sock_read=900))
        self.sink.emit("proxy.start", upstream=self.upstream, max_inflight=self.gate.limit,
                       pd_prefill=self.prefill, pd_decode=self.decode)

    async def stop(self, app: web.Application) -> None:
        if self.session:
            await self.session.close()
        self.sink.emit("proxy.stop")
        self.sink.close()

    def _fwd_headers(self, request: web.Request, rid: str) -> dict[str, str]:
        h = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        h["X-Request-Id"] = rid
        return h

    async def completion(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "body must be JSON"}, status=400)
        rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        if self.rid_in_body:  # SGLang takes its request id from the body, not a header
            body["rid"] = rid
        stream = bool(body.get("stream"))
        if stream:
            opts = body.get("stream_options") or {}
            opts["include_usage"] = True
            # First chunk then carries prompt + cached token counts (needs --enable-prompt-tokens-details
            # on vLLM, --enable-cache-report on SGLang): live prefix-cache hit/miss per request.
            opts["continuous_usage_stats"] = True
            body["stream_options"] = opts

        self.sink.emit(
            "req.arrive",
            rid=rid,
            endpoint=request.path,
            model=body.get("model"),
            stream=stream,
            max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
            workload=request.headers.get("X-Workload"),
            **_prompt_shape(body),
            **_prefix_family(body),
        )
        try:
            await self.gate.acquire()
        except asyncio.CancelledError:
            self.sink.emit("req.cancel", rid=rid, stage="queued")
            raise
        self.sink.emit("req.admit", rid=rid, inflight=self.gate.inflight, waiting=self.gate.waiting)
        replica = None
        try:
            url = self.upstream + request.path_qs
            headers = self._fwd_headers(request, rid)
            if self.replicas:
                replica = min(self.replicas, key=lambda u: self.replica_inflight[u])
                self.replica_inflight[replica] += 1
                url = replica + request.path_qs
                self.sink.emit("req.route", rid=rid, worker=replica, inflight=self.replica_inflight[replica])
            if self.prefill and self.decode:
                ktp, err = await self._remote_prefill(rid, request.path_qs, body, headers)
                if err is not None:
                    return err
                body = {**body, "kv_transfer_params": ktp}
                d_url = self._pick(self.decode, "d")
                url = d_url + request.path_qs
                self.sink.emit("req.decode_start", rid=rid, worker=d_url)
            if stream:
                return await self._stream(request, rid, url, body, headers)
            return await self._unary(rid, url, body, headers)
        except asyncio.CancelledError:
            self.sink.emit("req.cancel", rid=rid, stage="upstream")
            raise
        except Exception as e:  # upstream unreachable, protocol error, ...
            self.sink.emit("req.error", rid=rid, status=502, detail=f"{type(e).__name__}: {e}"[:500])
            return web.json_response({"error": str(e)}, status=502)
        finally:
            self.gate.release()
            if replica is not None:
                self.replica_inflight[replica] -= 1

    async def _unary(self, rid: str, url: str, body: dict, headers: dict) -> web.Response:
        assert self.session
        t0 = now_ns()
        async with self.session.post(url, json=body, headers=headers) as up:
            raw = await up.read()
            if up.status != 200:
                self.sink.emit("req.error", rid=rid, status=up.status, detail=raw[:500].decode(errors="replace"))
                return web.Response(status=up.status, body=raw, content_type=up.content_type)
            obj = json.loads(raw)
            chars, finish = 0, None
            for c in obj.get("choices") or []:
                msg = c.get("message") or {}
                chars += len(msg.get("content") or c.get("text") or "")
                finish = c.get("finish_reason") or finish
            self.sink.emit(
                "req.done",
                rid=rid,
                e2e_ms=(now_ns() - t0) / 1e6,
                chars=chars,
                finish_reason=finish,
                usage=obj.get("usage"),
                extras={k: v for k, v in obj.items() if k not in STANDARD_CHUNK_KEYS},
            )
            return web.Response(status=200, body=raw, content_type="application/json", headers={"X-Request-Id": rid})

    async def _stream(self, request: web.Request, rid: str, url: str, body: dict, headers: dict) -> web.StreamResponse:
        assert self.session
        t0 = now_ns()
        async with self.session.post(url, json=body, headers=headers) as up:
            if up.status != 200:
                raw = await up.read()
                self.sink.emit("req.error", rid=rid, status=up.status, detail=raw[:500].decode(errors="replace"))
                return web.Response(status=up.status, body=raw, content_type=up.content_type)

            resp = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Request-Id": rid},
            )
            await resp.prepare(request)

            buf = b""
            n_chunks = 0
            pending = 0
            last_emit = time.monotonic_ns()
            usage = None
            finish = None
            extras: dict[str, Any] = {}
            first_seen = False
            seen_usage = False
            try:
                async for data in up.content.iter_any():
                    await resp.write(data)
                    lines, buf = sse.iter_data_lines(buf + data)
                    for payload in lines:
                        if sse.needs_parse(payload, seen_usage):
                            obj = sse.parse(payload)
                            if obj:
                                _, f = _chunk_text_chars(obj)
                                finish = f or finish
                                if obj.get("usage"):
                                    usage = obj["usage"]
                                    if not seen_usage:
                                        seen_usage = True
                                        details = usage.get("prompt_tokens_details") or {}
                                        self.sink.emit("req.prefill_stats", rid=rid,
                                                       prompt_tokens=usage.get("prompt_tokens"),
                                                       cached_tokens=details.get("cached_tokens"),
                                                       t_ms=(now_ns() - t0) / 1e6)
                                for k, v in obj.items():
                                    if k not in STANDARD_CHUNK_KEYS:
                                        extras[k] = v
                        if not sse.has_text(payload):
                            continue
                        n_chunks += 1
                        pending += 1
                        now = time.monotonic_ns()
                        if not first_seen:
                            first_seen = True
                            self.sink.emit("req.first_token", rid=rid, ttft_ms=(now_ns() - t0) / 1e6)
                        # Token events are coalesced per request (~10 Hz): per-token events cost more
                        # Python time than the proxy can spare at high token rates.
                        if pending and (n_chunks == 1 or now - last_emit >= CHUNK_FLUSH_NS):
                            self.sink.emit("req.chunk", rid=rid, i=n_chunks - pending, n=pending)
                            pending = 0
                            last_emit = now
                if pending:
                    self.sink.emit("req.chunk", rid=rid, i=n_chunks - pending, n=pending)
                await resp.write_eof()
            except ConnectionResetError:
                self.sink.emit("req.cancel", rid=rid, stage="streaming", n_chunks=n_chunks)
                return resp

            self.sink.emit(
                "req.done",
                rid=rid,
                e2e_ms=(now_ns() - t0) / 1e6,
                n_chunks=n_chunks,
                finish_reason=finish,
                usage=usage,
                extras=extras,
            )
            return resp

    async def passthrough(self, request: web.Request) -> web.StreamResponse:
        assert self.session
        url = self.upstream + request.path_qs
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        data = await request.read()
        async with self.session.request(request.method, url, data=data or None, headers=headers) as up:
            raw = await up.read()
            return web.Response(status=up.status, body=raw, content_type=up.content_type)

    async def control(self, request: web.Request) -> web.Response:
        if request.method == "POST":
            body = await request.json()
            if "max_inflight" in body:
                v = body["max_inflight"]
                if v is not None and (not isinstance(v, int) or v < 1):
                    return web.json_response({"error": "max_inflight must be a positive int or null"}, status=400)
                old = self.gate.limit
                self.gate.set_limit(v)
                self.sink.emit("ctl.set", key="max_inflight", old=old, new=v)
        return web.json_response(
            {
                "upstream": self.upstream,
                "max_inflight": self.gate.limit,
                "inflight": self.gate.inflight,
                "waiting": self.gate.waiting,
                "events_dropped": self.sink.dropped,
            }
        )

    async def events(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        q = self.sink.subscribe()
        sender = asyncio.create_task(self._pump(ws, q))
        try:
            async for msg in ws:
                if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            sender.cancel()
            self.sink.unsubscribe(q)
        return ws

    @staticmethod
    async def _pump(ws: web.WebSocketResponse, q: asyncio.Queue) -> None:
        while True:
            ev = await q.get()
            await ws.send_str(json.dumps(ev, separators=(",", ":")))


def build_app(upstream: str, events_path: Path | None, max_inflight: int | None,
              prefill: list[str] | None = None, decode: list[str] | None = None) -> web.Application:
    sink = EventSink(events_path, src="proxy")
    proxy = Proxy(upstream, sink, max_inflight, prefill, decode)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["proxy"] = proxy
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_post("/v1/chat/completions", proxy.completion)
    app.router.add_post("/v1/completions", proxy.completion)
    app.router.add_route("*", "/_control", proxy.control)
    app.router.add_get("/_events", proxy.events)
    app.router.add_route("*", "/{tail:.*}", proxy.passthrough)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Timestamping, capped proxy in front of an inference engine")
    ap.add_argument("--upstream", required=True, help="engine base URL, e.g. http://127.0.0.1:8100")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--events", type=Path, help="append events to this JSONL file")
    ap.add_argument("--max-inflight", type=int, default=None, help="admission cap (default: none)")
    ap.add_argument("--pd-prefill", default="", help="comma-separated prefill instance URLs (enables P/D routing)")
    ap.add_argument("--pd-decode", default="", help="comma-separated decode instance URLs")
    ap.add_argument("--rid-in-body", action="store_true", help="also send the request id as body.rid (SGLang)")
    ap.add_argument("--replicas", default="", help="comma-separated independent replica URLs (least-inflight routing)")
    args = ap.parse_args()
    if args.events:
        args.events.parent.mkdir(parents=True, exist_ok=True)
    prefill = [u for u in args.pd_prefill.split(",") if u]
    decode = [u for u in args.pd_decode.split(",") if u]
    sse.install_uvloop()
    app = build_app(args.upstream, args.events, args.max_inflight, prefill, decode)
    app["proxy"].rid_in_body = args.rid_in_body
    if args.replicas:
        app["proxy"].set_replicas([u for u in args.replicas.split(",") if u])
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
