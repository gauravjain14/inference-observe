"""Turn a Mooncake production trace into a replayable workload.

Mooncake's open traces (github.com/kvcache-ai/Mooncake, FAST '25) list, per request: arrival time,
input length, output length, and `hash_ids`, one per 512-token block of the prompt. Requests that
share leading hash IDs share a prompt prefix. This tool gives every hash ID a fixed block of text,
so the replayed prompts share prefixes exactly where the original ones did, and the engine's prefix
cache sees the same reuse pattern.

  python -m harness.mooncake toolagent_trace.jsonl --out traces/mooncake-toolagent.jsonl \\
      --block-tokens 256 --duration 90 --stretch 3.5

  --block-tokens  tokens per hash block (512 in the original; smaller fits smaller GPUs)
  --duration      seconds of the original trace to take, from --start
  --stretch       slow arrivals down by this factor (3.5 = 3.5× fewer requests per second)
  --max-input     drop requests whose scaled prompt would exceed this many tokens
  --spread        the traces record arrivals at 3-second resolution (about 20 requests per
                  timestamp); spread each timestamp's requests evenly over that interval, in order
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Plain English words; each is about one token for common tokenizers.
WORDS = (
    "time person year way day thing man world life hand part child eye woman place work week case point "
    "government company number group problem fact be have do say get make go know take see come think look "
    "want give use find tell ask work seem feel try leave call good new first last long great little own "
    "other old right big high different small large next early young important few public bad same able "
    "to of in for on with at by from up about into over after beneath under above the and a that I it not "
    "he as you this but his they her she or an will my one all would there their what so up out if who "
    "system request response model token cache memory server client query result value record field table "
    "report status update event signal metric config option module service agent tool search document file"
).split()


def block_text(hash_id: int, tokens: int) -> str:
    """The same text for the same hash ID, every time; ends on a paragraph break so blocks join cleanly."""
    rng = random.Random(hash_id * 2654435761 % (2**32))
    n = max(8, tokens - 4)
    words = [rng.choice(WORDS) for _ in range(n)]
    for i in range(0, n, 12):
        words[i] = words[i].capitalize()
    return f"[{hash_id}] " + " ".join(words) + ".\n\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a Mooncake trace into a replayable workload")
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block-tokens", type=int, default=512)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the original trace")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--stretch", type=float, default=1.0)
    ap.add_argument("--max-input", type=int, default=14_000)
    ap.add_argument("--max-output", type=int, default=512)
    ap.add_argument("--workload", default="agentic", help="traffic class shown on the board")
    ap.add_argument("--spread", action="store_true", help="spread requests that share a timestamp over its interval")
    a = ap.parse_args()

    t0 = a.start * 1000
    t1 = t0 + a.duration * 1000
    rows = [json.loads(line) for line in open(a.trace) if line.strip()]
    rows = [r for r in rows if t0 <= r["timestamp"] < t1]
    if a.spread and rows:
        stamps = sorted({r["timestamp"] for r in rows})
        gaps = sorted(b - c for c, b in zip(stamps, stamps[1:]))
        res = gaps[len(gaps) // 2] if gaps else 0  # the trace's timestamp resolution
        group: dict[int, list[dict]] = {}
        for r in rows:
            group.setdefault(r["timestamp"], []).append(r)
        for ts, rs in group.items():
            for k, r in enumerate(rs):
                r["timestamp"] = ts + res * k / len(rs)
    scale = a.block_tokens / 512
    kept = dropped = 0
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            n_in = int(r["input_length"] * scale)
            if n_in > a.max_input:
                dropped += 1
                continue
            ids = r["hash_ids"]
            # the last block of the original prompt is usually partial: size it from input_length
            last = max(16, n_in - a.block_tokens * (len(ids) - 1))
            text = "".join(block_text(h, a.block_tokens if k < len(ids) - 1 else last) for k, h in enumerate(ids))
            req = {
                "id": f"{a.workload}-{i:05d}",
                "t": round((r["timestamp"] - t0) / 1000 * a.stretch, 3),
                "workload": a.workload,
                "body": {"messages": [{"role": "user", "content": text}],
                         "max_tokens": max(1, min(a.max_output, int(r["output_length"]))), "temperature": 0.7},
            }
            f.write(json.dumps(req) + "\n")
            kept += 1
    span = a.duration * a.stretch
    print(f"{kept} requests over {span:.0f} s ({kept / span:.2f}/s), {dropped} dropped (prompt > {a.max_input} tokens) -> {a.out}")


if __name__ == "__main__":
    main()
