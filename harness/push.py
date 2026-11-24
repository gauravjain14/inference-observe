"""Push the board stream out to the public relay (the Cloudflare Worker in relay-worker/).

Runs next to the collector and only connects outward. It reads the collector's stream, keeps what a newly
joining viewer needs, and sends the relay:
- "f" + the events of the last quarter second, as one batch;
- "s" + a snapshot of the current state, on connect and every 5 seconds.
The relay accepts the push with a token, read from a file so it never appears in a command line.

  io-push --relay wss://live.<domain>/ingest --token-file ~/.config/inference-observe/push-token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, WSMsgType

from .relay import StreamState

BATCH_S = 0.25
SNAPSHOT_S = 5.0
MAX_MESSAGE = 900_000  # bytes; the relay accepts up to 1 MiB per message
MAX_PENDING = 20_000  # events kept while the relay is unreachable; the next snapshot covers the rest


def read_token(path: Path) -> str:
    st = path.stat()
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(f"{path} must be readable only by you: chmod 600 {path}")
    token = path.read_text().strip()
    if len(token) < 32:
        raise SystemExit(f"{path} holds no usable token (32 characters or more)")
    return token


def batches(events: list[dict[str, Any]]) -> list[str]:
    """Events as one or more JSON arrays, each under MAX_MESSAGE."""
    out, cur, size = [], [], 2
    for e in events:
        s = json.dumps(e, separators=(",", ":"))
        if cur and size + len(s) + 1 > MAX_MESSAGE:
            out.append("[" + ",".join(cur) + "]")
            cur, size = [], 2
        cur.append(s)
        size += len(s) + 1
    if cur:
        out.append("[" + ",".join(cur) + "]")
    return out


class Push:
    def __init__(self, a: argparse.Namespace, token: str):
        self.a = a
        self.token = token
        self.state = StreamState()
        self.pending: list[dict[str, Any]] = []
        self.sent = {"batches": 0, "snapshots": 0, "bytes": 0, "connects": 0}

    async def upstream(self) -> None:
        while True:
            try:
                async with ClientSession() as s, s.ws_connect(self.a.upstream, heartbeat=20, max_msg_size=0) as ws:
                    async for msg in ws:
                        if msg.type == WSMsgType.TEXT:
                            evs, _ = self.state.take(msg.data)
                            self.pending.extend(evs)
                            if len(self.pending) > MAX_PENDING:
                                del self.pending[: len(self.pending) - MAX_PENDING]
            except Exception:
                pass
            await asyncio.sleep(2)

    def snapshot(self) -> str:
        evs = ([self.state.hello] if self.state.hello else []) + self.state.backlog()
        return "s" + json.dumps(evs, separators=(",", ":"))

    async def relay(self) -> None:
        backoff = 1.0
        while True:
            try:
                headers = {"Authorization": f"Bearer {self.token}"}
                async with ClientSession() as s, s.ws_connect(self.a.relay, headers=headers, heartbeat=20) as ws:
                    self.sent["connects"] += 1
                    backoff = 1.0
                    self.pending.clear()
                    snap = self.snapshot()
                    await ws.send_str(snap)
                    last_snap = time.monotonic()
                    while not ws.closed:
                        await asyncio.sleep(BATCH_S)
                        evs, self.pending = self.pending, []
                        for b in batches(evs):
                            await ws.send_str("f" + b)
                            self.sent["batches"] += 1
                            self.sent["bytes"] += len(b) + 1
                        if time.monotonic() - last_snap > SNAPSHOT_S:
                            snap = self.snapshot()
                            if len(snap) <= MAX_MESSAGE:
                                await ws.send_str(snap)
                                self.sent["snapshots"] += 1
                            last_snap = time.monotonic()
            except Exception as e:
                print(json.dumps({"t": time.strftime("%H:%M:%S"), "relay": "disconnected", "error": type(e).__name__}),
                      flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def report(self) -> None:
        while True:
            await asyncio.sleep(60)
            print(json.dumps({"t": time.strftime("%H:%M:%S"), **self.sent}), flush=True)

    async def run(self) -> None:
        await asyncio.gather(self.upstream(), self.relay(), self.report())


def main() -> None:
    ap = argparse.ArgumentParser(description="Push the board stream out to the public relay")
    ap.add_argument("--upstream", default="ws://127.0.0.1:18200/ws", help="the collector's board stream")
    ap.add_argument("--relay", required=True, help="the relay's ingest URL, e.g. wss://live.<domain>/ingest")
    ap.add_argument("--token-file", type=Path, default=Path(os.path.expanduser("~/.config/inference-observe/push-token")))
    a = ap.parse_args()
    asyncio.run(Push(a, read_token(a.token_file)).run())


if __name__ == "__main__":
    main()
