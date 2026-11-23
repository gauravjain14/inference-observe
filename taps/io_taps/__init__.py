"""Engine taps for inference-observe. Each tap sends small JSON datagrams over UDP to the
collector (IO_TAP_HOST:IO_TAP_PORT, default 127.0.0.1:18201). Sends never block: if the
socket buffer is full the datagram is dropped and counted, so a tap cannot slow the engine."""

import json
import os
import socket
import time
from typing import Any


class UdpSender:
    def __init__(self, src: str):
        self.src = src
        self.instance = os.environ.get("IO_TAP_INSTANCE", "engine")
        self.url = os.environ.get("IO_TAP_URL")
        self.addr = (os.environ.get("IO_TAP_HOST", "127.0.0.1"), int(os.environ.get("IO_TAP_PORT", "18201")))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.addr[0].startswith("127.") or self.addr[0] == "localhost":
            self.sock.bind(("127.0.0.1", 0))  # a local collector: the send socket is not reachable from the network
        self.sock.setblocking(False)
        self.sent = 0
        self.dropped = 0

    def send(self, obj: dict[str, Any]) -> None:
        obj.setdefault("v", 0)
        obj.setdefault("t_ns", time.time_ns())
        obj.setdefault("src", self.src)
        obj.setdefault("instance", self.instance)
        try:
            data = json.dumps(obj, separators=(",", ":"), default=str).encode()
            if len(data) > 60_000:  # keep under the UDP datagram limit
                obj.pop("finished", None)
                obj["truncated"] = True
                data = json.dumps(obj, separators=(",", ":"), default=str).encode()
            self.sock.sendto(data, self.addr)
            self.sent += 1
        except (BlockingIOError, OSError):
            self.dropped += 1
