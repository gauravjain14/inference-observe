"""Cheap scanning of OpenAI-style SSE lines.

The proxy and replayer sit on the hot path of every streamed token. Parsing each chunk as
JSON costs enough Python time to saturate one event loop at a few thousand tokens/s (it did:
the harness, not the engine, was the bottleneck at ~5.5k tok/s). These helpers use byte
searches for the per-token decisions and parse JSON only on the rare lines that carry
usage, finish_reason or engine timing extras.
"""

from __future__ import annotations

import json
from typing import Any

_TEXT_KEYS = (b'"content":', b'"text":', b'"reasoning_content":', b'"reasoning":')


def has_text(line: bytes) -> bool:
    """True if this data line carries a non-empty text delta (content, text or reasoning)."""
    for key in _TEXT_KEYS:
        i = line.find(key)
        while i >= 0:
            j = i + len(key)
            while j < len(line) and line[j] == 0x20:  # optional space after the colon
                j += 1
            if j + 1 < len(line) and line[j] == 0x22 and line[j + 1] != 0x22:  # opening quote, not empty ""
                return True
            i = line.find(key, j)
    return False


def has_usage(line: bytes) -> bool:
    return b'"usage":{' in line or b'"usage": {' in line


def is_final(line: bytes) -> bool:
    """The closing chunks: the one with finish_reason, and the usage-only chunk (choices: [])."""
    return (b'"choices":[]' in line or b'"choices": []' in line or b'"finish_reason":"' in line
            or b'"finish_reason": "' in line)


def needs_parse(line: bytes, seen_usage: bool = True) -> bool:
    """Lines worth a full JSON parse.

    With continuous usage stats every chunk carries usage, so usage alone only triggers a parse
    until the first one has been seen (it holds prompt and cached token counts). After that, only
    closing chunks and engine extras (P/D kv_transfer_params) are parsed.
    """
    if not seen_usage and has_usage(line):
        return True
    return is_final(line) or b'"timings"' in line or b'"kv_transfer_params"' in line


def parse(payload: bytes) -> dict[str, Any] | None:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def iter_data_lines(buf: bytes) -> tuple[list[bytes], bytes]:
    """Split a buffer into complete `data:` payloads and the unfinished remainder."""
    if b"\n" not in buf:
        return [], buf
    *lines, rest = buf.split(b"\n")
    out = []
    for line in lines:
        line = line.strip()
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            if payload and payload != b"[DONE]":
                out.append(payload)
    return out, rest


def install_uvloop() -> None:
    try:
        import asyncio

        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
