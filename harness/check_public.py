"""Check a public deployment from the outside: the feed is read-only and nothing else is reachable.

  io-check-public wss://live.<domain>/ws --origin https://<user>.github.io \\
      --page https://<user>.github.io/<repo>/

Run it from any machine after the tunnel and the page are up. Every line should read PASS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import socket
from urllib.parse import urlsplit

import aiohttp

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


async def refused(s: aiohttp.ClientSession, method: str, url: str, origin: str) -> str:
    try:
        async with s.request(method, url, headers={"Origin": origin}, allow_redirects=False) as r:
            return str(r.status)
    except Exception as e:
        return type(e).__name__


async def closes(ws: aiohttp.ClientWebSocketResponse) -> int | None:
    for _ in range(500):
        m = await ws.receive(timeout=10)
        if m.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
            return ws.close_code
    return None


async def run(a: argparse.Namespace) -> None:
    feed = urlsplit(a.feed)
    base = f"{'https' if feed.scheme == 'wss' else 'http'}://{feed.netloc}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        ws = await s.ws_connect(a.feed, headers={"Origin": a.origin})
        first = json.loads((await ws.receive(timeout=15)).data)
        check("the page's origin can watch", first.get("type") == "hello", f"{first.get('engine')} {first.get('model')}")
        await ws.close()
        for headers, name in (({"Origin": "https://evil.example"}, "another site"), ({}, "no origin")):
            try:
                w = await s.ws_connect(a.feed, headers=headers)
                await w.close()
                check(f"{name} is refused", False, "connected")
            except aiohttp.WSServerHandshakeError as e:
                check(f"{name} is refused", e.status == 403, f"HTTP {e.status}")
        for method, path in (("GET", "/"), ("GET", "/note"), ("POST", "/note"), ("GET", "/health"), ("POST", "/ws"),
                             ("POST", "/v1/completions"), ("POST", "/v1/chat/completions"), ("GET", "/v1/models"),
                             ("GET", "/metrics"), ("GET", "/ws/"), ("GET", "/_events")):
            st = await refused(s, method, base + path, a.origin)
            check(f"{method} {path} is not served", st in ("400", "403", "404", "405"), f"HTTP {st}")
        ws = await s.ws_connect(a.feed, headers={"Origin": a.origin})
        await ws.send_str('{"type":"advisor.note","id":"probe","title":"probe"}')
        code = await closes(ws)
        check("a viewer that sends data is disconnected", code == 1008, f"close code {code}")
    # one address family, so every connection comes from the same address
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(family=socket.AF_INET, force_close=True)) as s4:
        conns: list = []
        for _ in range(a.max_per_ip + 1):
            try:
                conns.append(await s4.ws_connect(a.feed, headers={"Origin": a.origin}))
            except aiohttp.WSServerHandshakeError as e:
                conns.append(e.status)
        check(f"more than {a.max_per_ip} viewers from one address are refused", conns[-1] == 429,
              f"{['ok' if not isinstance(c, int) else c for c in conns]}")
        for c in conns:
            if not isinstance(c, int):
                await c.close()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        if a.page:
            async with s.get(a.page) as r:
                html = await r.text()
            csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html)
            check("the page carries its Content-Security-Policy", bool(csp), "")
            if csp:
                pol = csp.group(1)
                check("the policy allows only its own scripts", "script-src 'self'" in pol and "unsafe-eval" not in pol, "")
                check("the policy allows only this feed", f"{feed.scheme}://{feed.netloc}" in pol, re.search(r"connect-src [^;]+", pol).group(0))
    passed = sum(ok for ok, *_ in results)
    print(f"\n{passed}/{len(results)} passed")
    raise SystemExit(0 if passed == len(results) else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check a public deployment from the outside")
    ap.add_argument("feed", help="the public feed, e.g. wss://live.<domain>/ws")
    ap.add_argument("--origin", required=True, help="the page's origin, e.g. https://<user>.github.io")
    ap.add_argument("--page", help="the page URL, to check its Content-Security-Policy")
    ap.add_argument("--max-per-ip", type=int, default=4, help="the relay's --max-per-ip")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
