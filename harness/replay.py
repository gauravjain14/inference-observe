"""Replays a recorded traffic trace against an OpenAI-compatible endpoint (normally the proxy).

Each trace line is {"id", "t" (seconds from start), "workload", "endpoint"?, "body"}.
Requests are sent at their scheduled times regardless of how earlier ones are doing,
so overload shows up as queueing instead of being hidden by a closed loop.

Writes client-side events to <out>/client.jsonl and a summary to <out>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import uuid
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, TCPConnector

from . import sse
from .events import EventSink, now_ns


def load_trace(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        reqs = [json.loads(line) for line in f if line.strip()]
    return sorted(reqs, key=lambda r: r["t"])


async def detect_model(session: ClientSession, base: str) -> str | None:
    try:
        async with session.get(f"{base}/v1/models") as r:
            data = await r.json(content_type=None)
            models = data.get("data") or data.get("models") or []
            if not models:
                return None
            return models[0].get("id") or models[0].get("name")
    except Exception:
        return None


def salted(messages: Any, nonce: str) -> Any:
    """Make the first user message unique per replay, after any system prompt.

    Replaying one trace in a loop would otherwise resend identical prompts, and prefix caching
    would serve them almost entirely from cache: unrealistically good hit rates. Shared system
    prompts (agent sessions) stay shared because they come before the first user message.
    """
    if not isinstance(messages, list):
        return messages
    out = [dict(m) if isinstance(m, dict) else m for m in messages]
    for m in out:
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = f"[{nonce}] " + m["content"]
            break
    return out


async def send_one(session: ClientSession, base: str, req: dict[str, Any], model: str | None, sink: EventSink,
                   lateness_ms: float, nonce: str, salt: bool = True) -> dict[str, Any]:
    # Unique per replay: the same trace replayed twice into one run must not share request ids,
    # or per-request joins (proxy <-> engine <-> client) silently mix the two replays.
    rid = f"{req['id']}~{nonce}"
    body = dict(req["body"])
    if salt:
        body["messages"] = salted(body.get("messages"), nonce)
    body["stream"] = True
    if model:
        body["model"] = model
    endpoint = req.get("endpoint", "/v1/chat/completions")
    headers = {"X-Request-Id": rid, "X-Workload": req.get("workload", "")}
    t0 = now_ns()
    sink.emit("client.send", rid=rid, scheduled_t=req["t"], lateness_ms=lateness_ms)
    result: dict[str, Any] = {"rid": rid, "workload": req.get("workload"), "ok": False}
    try:
        async with session.post(base + endpoint, json=body, headers=headers) as r:
            if r.status != 200:
                detail = (await r.text())[:300]
                sink.emit("client.error", rid=rid, status=r.status, detail=detail)
                return {**result, "status": r.status, "detail": detail}
            ttft_ms = None
            n_chunks = 0
            usage = None
            buf = b""
            async for data in r.content.iter_any():
                lines, buf = sse.iter_data_lines(buf + data)
                for payload in lines:
                    if sse.is_final(payload):
                        obj = sse.parse(payload)
                        if obj and obj.get("usage"):
                            usage = obj["usage"]
                    if sse.has_text(payload):
                        if ttft_ms is None:
                            ttft_ms = (now_ns() - t0) / 1e6
                            sink.emit("client.first_token", rid=rid, ttft_ms=ttft_ms)
                        n_chunks += 1
            e2e_ms = (now_ns() - t0) / 1e6
            sink.emit("client.done", rid=rid, e2e_ms=e2e_ms, n_chunks=n_chunks, usage=usage)
            return {**result, "ok": True, "ttft_ms": ttft_ms, "e2e_ms": e2e_ms, "n_chunks": n_chunks, "usage": usage}
    except Exception as e:
        sink.emit("client.error", rid=rid, status=None, detail=f"{type(e).__name__}: {e}"[:300])
        return {**result, "detail": f"{type(e).__name__}: {e}"}


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return round(xs[k], 1)


def summarize(results: list[dict[str, Any]], wall_s: float, max_lateness_ms: float) -> dict[str, Any]:
    ok = [r for r in results if r["ok"]]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    e2e = [r["e2e_ms"] for r in ok]
    out_tokens = [((r.get("usage") or {}).get("completion_tokens")) for r in ok]
    known = [t for t in out_tokens if isinstance(t, int)]
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "wall_s": round(wall_s, 2),
        "ttft_ms": {"p50": pct(ttft, 50), "p90": pct(ttft, 90), "p99": pct(ttft, 99)},
        "e2e_ms": {"p50": pct(e2e, 50), "p90": pct(e2e, 90), "p99": pct(e2e, 99)},
        "output_tokens": sum(known) if known else None,
        "output_tok_per_s": round(sum(known) / wall_s, 1) if known and wall_s > 0 else None,
        "usage_reported": f"{len(known)}/{len(ok)}",
        "max_send_lateness_ms": round(max_lateness_ms, 1),
        "first_errors": [r.get("detail") for r in results if not r["ok"]][:3],
    }


async def run(trace: Path, base: str, out: Path, speed: float, limit: int | None, model: str | None,
              salt: bool = True) -> dict[str, Any]:
    reqs = load_trace(trace)[:limit] if limit else load_trace(trace)
    out.mkdir(parents=True, exist_ok=True)
    sink = EventSink(out / "client.jsonl", src="client")
    # limit=0: aiohttp's default pool of 100 connections would queue requests client-side and
    # silently cap the load at 100 concurrent requests.
    async with ClientSession(connector=TCPConnector(limit=0),
                             timeout=ClientTimeout(total=None, sock_connect=10, sock_read=900)) as session:
        model = model or await detect_model(session, base)
        nonce = uuid.uuid4().hex[:6]
        sink.emit("client.start", trace=str(trace), base=base, speed=speed, n=len(reqs), model=model, nonce=nonce)
        loop = asyncio.get_running_loop()
        start = loop.time()
        tasks = []
        max_late = 0.0
        for req in reqs:
            due = start + req["t"] / speed
            delay = due - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            late_ms = max(0.0, (loop.time() - due) * 1000)
            max_late = max(max_late, late_ms)
            tasks.append(asyncio.create_task(send_one(session, base, req, model, sink, late_ms, nonce, salt)))
        results = await asyncio.gather(*tasks)
        wall = loop.time() - start
    summary = summarize(list(results), wall, max_late)
    sink.emit("client.summary", **summary)
    sink.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a traffic trace at its recorded arrival times")
    ap.add_argument("trace", type=Path)
    ap.add_argument("--base", default="http://127.0.0.1:18000", help="proxy (or engine) base URL")
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--speed", type=float, default=1.0, help="time scale: 2.0 replays twice as fast")
    ap.add_argument("--limit", type=int, help="only the first N requests")
    ap.add_argument("--model", help="model name to send (default: first model the server lists)")
    ap.add_argument("--no-salt", action="store_true",
                    help="send prompts verbatim (repeat replays then hit the prefix cache unrealistically)")
    args = ap.parse_args()
    sse.install_uvloop()
    summary = asyncio.run(run(args.trace, args.base.rstrip("/"), args.out, args.speed, args.limit, args.model,
                              salt=not args.no_salt))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
