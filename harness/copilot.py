"""Replay GitHub Copilot coding-agent sessions against an OpenAI-compatible endpoint (normally the proxy).

Source: "GitHub Copilot Coding Agent Traces 2026", Azure Public Dataset (CC-BY), Liu et al.,
"Agentic Coding in the Wild", arXiv 2608.00101. Each record is one agent session: user turns, each a
loop of model calls and tool runs, with timestamps, durations and prompt, completion and cached token
counts. The trace contains no text.

The replay is closed-loop, one task per session:
- a session starts at its recorded time;
- each call waits for the call it followed in production, then for the recorded gap (a tool run or a
  user pause), so the engine's own speed sets the pace;
- each prompt is the session's conversation so far (the previous prompt and the engine's own answer)
  plus filler text for the new content (tool output, the user's next message), sized to the recorded
  prompt; a prompt much shorter than the conversation (context compaction) keeps only the system part;
- each answer is generated to the recorded length (max_tokens with ignore_eos).

The engine's prefix cache decides what is reused. Production's cached count is written next to the
engine's for comparison. Traffic classes on the board: new session, user turn, agent loop, and
session in progress for the first call of a session that began before the replayed window (its cache
starts empty).

  io-copilot downloaded_data/date=2026-06-06 --start 2026-06-06T13:00 --duration 14400 \\
      --sample 0.12 --prompt-scale 0.125 --out runs/live --loop
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import gzip
import hashlib
import json
import random
import signal
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, TCPConnector

from . import sse
from .events import EventSink, now_ns
from .mooncake import WORDS
from .replay import detect_model

NEW_MIN = 32  # tokens of new content when a prompt shrinks slightly (changed IDE state at the end)


@dataclass
class Call:
    t: float  # production start, Unix seconds
    end: float  # production end
    prompt: int
    completion: int
    cached: int
    initiator: str  # "user" starts a turn, "agent" follows a tool run
    first: bool  # first call of the production session
    system: int  # production tokens in the System segment, 0 if not reported


@dataclass
class Session:
    sid: str
    calls: list[Call]
    ctx: list[list[Any]] = field(default_factory=list)  # [text, tokens] pieces of the conversation
    pieces: int = 0

    def tokens(self) -> int:
        return sum(p[1] for p in self.ctx)

    def keep(self, n: int) -> None:
        """Cut the conversation to its first n tokens, on a word boundary."""
        out, total = [], 0
        for text, k in self.ctx:
            if total + k <= n:
                out.append([text, k])
                total += k
                continue
            room = n - total
            if room > 0 and k > 0:
                words = text.split(" ")
                cut = max(1, int(len(words) * room / k))
                out.append([" ".join(words[:cut]) + " ", room])
            break
        self.ctx = out


def parse_ts(s: str) -> float:
    base = dt.datetime.fromisoformat(s[:19]).replace(tzinfo=dt.timezone.utc).timestamp()
    frac = s[20:29].rstrip("Z") if len(s) > 20 and s[19] == "." else ""
    return base + (float("0." + frac) if frac else 0.0)


def sampled(sid: str, fraction: float, seed: str) -> bool:
    h = int.from_bytes(hashlib.blake2b((seed + sid).encode(), digest_size=8).digest(), "big")
    return h / 2**64 < fraction


def load(src: Path, t0: float, t1: float, fraction: float, seed: str, pscale: float,
         max_prompt: int) -> tuple[list[Session], dict[str, int]]:
    files = sorted(src.glob("*.jsonl.gz")) if src.is_dir() else [src]
    out: list[Session] = []
    stats = {"sessions_seen": 0, "too_long": 0}
    for f in files:
        with gzip.open(f, "rt") if f.suffix == ".gz" else open(f) as fh:
            for line in fh:
                s = json.loads(line)
                stats["sessions_seen"] += 1
                if not sampled(s["session_id"], fraction, seed):
                    continue
                calls = []
                for turn in s.get("turns") or []:
                    for c in turn.get("llm_calls") or []:
                        tk = c.get("tokens") or {}
                        if not tk.get("prompt") or not c.get("timestamp"):
                            continue
                        t = parse_ts(c["timestamp"])
                        system = sum(m.get("token_len") or 0 for m in c.get("message_metadata") or []
                                     if m.get("type") == "System")
                        calls.append(Call(t, t + (c.get("duration_ms") or 0) / 1000, tk["prompt"],
                                          tk.get("completion") or 0, tk.get("cached") or 0,
                                          c.get("initiator_type") or "agent", False, system))
                if not calls:
                    continue
                calls.sort(key=lambda c: c.t)
                calls[0].first = True
                calls = [c for c in calls if t0 <= c.t < t1]
                if not calls:
                    continue
                if max(c.prompt for c in calls) * pscale > max_prompt:
                    stats["too_long"] += 1
                    continue
                out.append(Session(s["session_id"], calls))
    out.sort(key=lambda s: s.calls[0].t)
    return out, stats


def call_class(c: Call) -> str:
    return "new session" if c.first else "user turn" if c.initiator == "user" else "agent loop"


class Replay:
    def __init__(self, a: argparse.Namespace, sink: EventSink, log: Any):
        self.a = a
        self.sink = sink
        self.log = log
        self.tok_per_word = 1.0  # calibrated from the engine's reported prompt tokens
        self.done: list[dict[str, Any]] = []
        self.active = 0
        self.errors = 0

    def filler(self, sess: Session, tokens: int) -> str:
        sess.pieces += 1
        rng = random.Random(f"{self.nonce}/{sess.sid}/{sess.pieces}")
        n = max(1, round(tokens / self.tok_per_word))
        words = [rng.choice(WORDS) for _ in range(n)]
        for i in range(0, n, 14):
            words[i] = words[i].capitalize()
        return " ".join(words) + ".\n"

    def build(self, sess: Session, c: Call) -> tuple[str, int, int]:
        a = self.a
        p = max(16, round(c.prompt * a.prompt_scale))
        o = max(1, min(a.max_output, round(c.completion * a.output_scale)))
        base = sess.tokens()
        if base and p < base * 0.5:  # context compaction: a new conversation after the system part
            sess.keep(min(round(c.system * a.prompt_scale), p - NEW_MIN, base))
        elif base and p < base:
            sess.keep(p - NEW_MIN)
        new = p - sess.tokens()
        if new > 0:
            sess.ctx.append([self.filler(sess, new), new])
        return "".join(t for t, _ in sess.ctx), p, o

    async def call(self, http: ClientSession, sess: Session, k: int, c: Call, joined: bool) -> None:
        prompt, p, o = self.build(sess, c)
        rid = f"cp-{sess.sid[:8]}-{k:03d}~{self.nonce}"
        body = {"model": self.model, "prompt": prompt, "max_tokens": o, "temperature": 0.7, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True}}
        cls = "session in progress" if joined else call_class(c)
        headers = {"X-Request-Id": rid, "X-Workload": cls}
        t0 = now_ns()
        self.sink.emit("client.send", rid=rid, cls=cls, prompt_target=p, max_tokens=o)
        rec: dict[str, Any] = {"rid": rid, "cls": cls, "prod_prompt": c.prompt, "prod_cached": c.cached,
                               "prompt_target": p, "max_tokens": o, "ok": False}
        text: list[str] = []
        try:
            async with http.post(self.a.base + "/v1/completions", json=body, headers=headers) as r:
                if r.status != 200:
                    rec["detail"] = (await r.text())[:300]
                    raise RuntimeError(rec["detail"])
                buf = b""
                async for data in r.content.iter_any():
                    lines, buf = sse.iter_data_lines(buf + data)
                    for payload in lines:
                        obj = sse.parse(payload)
                        if not obj:
                            continue
                        for ch in obj.get("choices") or []:
                            if ch.get("text"):
                                if "ttft_ms" not in rec:
                                    rec["ttft_ms"] = (now_ns() - t0) / 1e6
                                text.append(ch["text"])
                        if obj.get("usage"):
                            rec["usage"] = obj["usage"]
            rec["ok"] = True
        except Exception as e:  # keep the session going; the board shows the error
            self.errors += 1
            rec.setdefault("detail", f"{type(e).__name__}: {e}"[:300])
        rec["e2e_ms"] = (now_ns() - t0) / 1e6
        u = rec.get("usage") or {}
        if u.get("prompt_tokens"):
            rec["prompt_tokens"] = u["prompt_tokens"]
            rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            ratio = u["prompt_tokens"] / max(1, p)
            self.tok_per_word = min(2.0, max(0.5, self.tok_per_word * (0.9 + 0.1 * ratio)))
        if text:
            sess.ctx.append(["".join(text), u.get("completion_tokens") or len(text)])
        self.sink.emit("client.done", **{k: v for k, v in rec.items() if k != "usage"})
        self.done.append(rec)

    async def session(self, http: ClientSession, sess: Session, start: float, t0: float) -> None:
        loop = asyncio.get_running_loop()
        speed = self.a.speed
        starts: dict[int, float] = {}
        ends: dict[int, float] = {}
        done: dict[int, asyncio.Event] = {}
        tasks = []
        self.active += 1
        try:
            for k, c in enumerate(sess.calls):
                before = [j for j in range(k) if sess.calls[j].end <= c.t]
                if before:  # follows a finished call: wait for it, then the recorded gap
                    j = max(before, key=lambda j: sess.calls[j].end)
                    await done[j].wait()
                    due = ends[j] + (c.t - sess.calls[j].end) / speed
                elif k:  # overlaps the previous call in production: same offset from its start
                    due = starts[k - 1] + (c.t - sess.calls[k - 1].t) / speed
                else:
                    due = start + (c.t - t0) / speed
                await asyncio.sleep(max(0.0, due - loop.time()))
                starts[k] = loop.time()
                done[k] = asyncio.Event()
                joined = k == 0 and not c.first  # the session began before the window: its cache starts empty
                tasks.append(asyncio.create_task(self._run(http, sess, k, c, joined, ends, done[k])))
            await asyncio.gather(*tasks)
        finally:
            self.active -= 1

    async def _run(self, http, sess, k, c, joined, ends, ev) -> None:
        try:
            await self.call(http, sess, k, c, joined)
        finally:
            ends[k] = asyncio.get_running_loop().time()
            ev.set()

    async def report(self) -> None:
        last = 0
        while True:
            await asyncio.sleep(self.a.report_every)
            recent = self.done[last:]
            last = len(self.done)
            print(json.dumps({"t": time.strftime("%H:%M:%S"), "sessions_active": self.active,
                              "calls_done": len(self.done), "errors": self.errors, **window_stats(recent)}),
                  file=self.log, flush=True)

    async def run(self, sessions: list[Session], t0: float) -> None:
        async with ClientSession(connector=TCPConnector(limit=0),
                                 timeout=ClientTimeout(total=None, sock_connect=10, sock_read=900)) as http:
            self.model = self.a.model or await detect_model(http, self.a.base)
            rep = asyncio.create_task(self.report())
            n = 0
            while True:
                n += 1
                self.nonce = uuid.uuid4().hex[:6]
                for s in sessions:
                    s.ctx, s.pieces = [], 0
                self.sink.emit("client.start", source="copilot", pass_n=n, sessions=len(sessions),
                               calls=sum(len(s.calls) for s in sessions), model=self.model, nonce=self.nonce)
                start = asyncio.get_running_loop().time()
                await asyncio.gather(*(self.session(http, s, start, t0) for s in sessions))
                if not self.a.loop:
                    break
            rep.cancel()


def window_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    every = [r for r in recs if r.get("prompt_tokens")]
    ok = [r for r in every if r["cls"] != "session in progress"]  # compared with production
    if not ok:
        return {}
    out: dict[str, Any] = {
        "calls": len(ok),
        "cached_share": round(sum(r["cached_tokens"] for r in ok) / sum(r["prompt_tokens"] for r in ok), 3),
        "production_cached_share": round(sum(r["prod_cached"] for r in ok) / sum(r["prod_prompt"] for r in ok), 3),
        "ttft_ms_p50": round(statistics.median(r["ttft_ms"] for r in ok if "ttft_ms" in r), 1)
        if any("ttft_ms" in r for r in ok) else None,
    }
    for cls in ("new session", "user turn", "agent loop", "session in progress"):
        rs = [r for r in every if r["cls"] == cls]
        if rs:
            out[cls] = {"n": len(rs),
                        "cached": round(sum(r["cached_tokens"] for r in rs) / sum(r["prompt_tokens"] for r in rs), 3),
                        "production": round(sum(r["prod_cached"] for r in rs) / sum(r["prod_prompt"] for r in rs), 3)}
    return out


def plan(sessions: list[Session], a: argparse.Namespace, t0: float, t1: float) -> dict[str, Any]:
    calls = [c for s in sessions for c in s.calls]
    mins = max(1.0, (t1 - t0) / 60)
    ps = sorted(round(c.prompt * a.prompt_scale) for c in calls) or [0]
    return {"sessions": len(sessions), "calls": len(calls), "calls_per_min": round(len(calls) / mins * a.speed, 1),
            "prompt_tokens_p50": ps[len(ps) // 2], "prompt_tokens_p90": ps[int(len(ps) * 0.9)],
            "prompt_tokens_max": ps[-1]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay GitHub Copilot agent sessions, closed-loop")
    ap.add_argument("src", type=Path, help="a date=YYYY-MM-DD folder of shards, or one shard")
    ap.add_argument("--base", default="http://127.0.0.1:18000", help="proxy (or engine) base URL")
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--start", required=True, help="UTC start of the window, e.g. 2026-06-06T14:00")
    ap.add_argument("--duration", type=float, default=3600, help="seconds of the trace to replay")
    ap.add_argument("--sample", type=float, default=0.12, help="fraction of sessions to replay")
    ap.add_argument("--seed", default="io", help="session sample seed")
    ap.add_argument("--prompt-scale", type=float, default=0.125, help="prompt tokens × this")
    ap.add_argument("--output-scale", type=float, default=1.0, help="completion tokens × this")
    ap.add_argument("--max-prompt", type=int, default=30_000, help="skip sessions whose scaled prompt exceeds this")
    ap.add_argument("--max-output", type=int, default=1024)
    ap.add_argument("--speed", type=float, default=1.0, help="time scale: 2.0 replays twice as fast")
    ap.add_argument("--model", help="model name to send (default: first model the server lists)")
    ap.add_argument("--loop", action="store_true", help="replay the window again when it ends")
    ap.add_argument("--dry-run", action="store_true", help="print the load plan and exit")
    ap.add_argument("--report-every", type=float, default=30.0)
    a = ap.parse_args()
    a.base = a.base.rstrip("/")
    t0 = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc).timestamp()
    t1 = t0 + a.duration
    sessions, stats = load(a.src, t0, t1, a.sample, a.seed, a.prompt_scale, a.max_prompt)
    print(json.dumps({**stats, **plan(sessions, a, t0, t1)}), flush=True)
    if a.dry_run:
        return
    a.out.mkdir(parents=True, exist_ok=True)
    sink = EventSink(a.out / "client.jsonl", src="client")
    sse.install_uvloop()
    r = Replay(a, sink, log=open(a.out / "copilot.log", "a"))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        asyncio.run(r.run(sessions, t0))
    except KeyboardInterrupt:
        pass
    finally:
        summary = {"calls": len(r.done), "errors": r.errors, **window_stats(r.done)}
        (a.out / "copilot-summary.json").write_text(json.dumps(summary, indent=2))
        with open(a.out / "copilot-calls.jsonl", "w") as f:
            for rec in r.done:
                f.write(json.dumps({k: v for k, v in rec.items() if k != "usage"}) + "\n")
        sink.close()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
