"""Deterministic traffic traces for engine comparisons.

Five workloads, each meant to light up a different part of an engine:
  chat     short prompts, medium outputs, steady arrivals        (baseline decode)
  rag      3-6k-token prompts, short outputs                      (prefill, chunked prefill)
  agentic  one 2k-token shared system prompt, multi-turn sessions (prefix caching)
  burst    steady load with a 10 s spike far above capacity       (queueing, KV pressure, preemption)
  spec     near-copy rewriting tasks, natural stopping            (speculative decoding acceptance)

Text comes from MATH-500 (problems as questions, solutions as filler), cut to target
token lengths with the model's tokenizer. Same seed + same files = same trace.
Output lengths are fixed with max_tokens + ignore_eos (except "spec") so every engine
does the same amount of decode work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

from tokenizers import Tokenizer

HF_HUB = Path.home() / ".cache/huggingface/hub"
DEFAULT_TOKENIZER = "models--Qwen--Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET = "datasets--HuggingFaceH4--MATH-500"


def find_snapshot_file(repo_dir: str, name: str) -> Path:
    hits = sorted((HF_HUB / repo_dir / "snapshots").glob(f"*/{name}"))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {HF_HUB / repo_dir}")
    return hits[0]


class TextPool:
    def __init__(self, dataset: Path, tokenizer: Tokenizer):
        rows = [json.loads(line) for line in open(dataset) if line.strip()]
        self.questions = [r["problem"].strip() for r in rows]
        corpus = "\n\n".join(r["solution"].strip() for r in rows)
        self.tok = tokenizer
        self.ids = tokenizer.encode(corpus).ids

    def passage(self, rng: random.Random, n_tokens: int) -> str:
        start = rng.randrange(0, len(self.ids) - n_tokens - 1)
        return self.tok.decode(self.ids[start : start + n_tokens])

    def fixed_passage(self, offset: int, n_tokens: int) -> str:
        return self.tok.decode(self.ids[offset : offset + n_tokens])

    def ntok(self, text: str) -> int:
        return len(self.tok.encode(text).ids)


def poisson_times(rng: random.Random, rate: float, duration: float, start: float = 0.0) -> list[float]:
    ts, t = [], start
    while True:
        t += rng.expovariate(rate)
        if t >= start + duration:
            return ts
        ts.append(round(t, 4))


def req(workload: str, i: int, t: float, messages: list[dict[str, str]], max_tokens: int, ignore_eos: bool = True,
        **meta: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if ignore_eos:
        body["ignore_eos"] = True
    return {"id": f"{workload}-{i:04d}", "t": t, "workload": workload, "body": body, "meta": meta}


def chat(pool: TextPool, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for i, t in enumerate(poisson_times(rng, rate=2.0, duration=60)):
        q = rng.choice(pool.questions)
        msgs = [{"role": "user", "content": f"Solve this problem and explain each step.\n\n{q}"}]
        out.append(req("chat", i, t, msgs, rng.randint(128, 384)))
    return out


def rag(pool: TextPool, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for i, t in enumerate(poisson_times(rng, rate=0.5, duration=60)):
        n_ctx = rng.randint(3000, 6000)
        notes = pool.passage(rng, n_ctx)
        q = rng.choice(pool.questions)
        msgs = [{"role": "user", "content": f"Use the reference notes to answer.\n\n<notes>\n{notes}\n</notes>\n\nQuestion: {q}"}]
        out.append(req("rag", i, t, msgs, rng.randint(64, 128), context_tokens=n_ctx))
    return out


def agentic(pool: TextPool, rng: random.Random) -> list[dict[str, Any]]:
    system = (
        "You are a careful assistant with access to the following reference manual. "
        "Answer using it where relevant.\n\n" + pool.fixed_passage(offset=1000, n_tokens=2000)
    )
    out = []
    i = 0
    for s in range(8):
        t = rng.uniform(0, 20)
        history: list[dict[str, str]] = []
        for turn in range(6):
            q = rng.choice(pool.questions)
            msgs = [{"role": "system", "content": system}, *history, {"role": "user", "content": q}]
            out.append(req("agentic", i, round(t, 4), msgs, rng.randint(96, 192), session=s, turn=turn))
            i += 1
            # Stand-in for the previous answer so later turns share a growing, cacheable prefix.
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": pool.passage(rng, 150)}]
            t += rng.uniform(3, 6)
    out.sort(key=lambda r: r["t"])
    return out


def burst(pool: TextPool, rng: random.Random) -> list[dict[str, Any]]:
    times = sorted(poisson_times(rng, rate=1.0, duration=60) + poisson_times(rng, rate=12.0, duration=10, start=20))
    out = []
    for i, t in enumerate(times):
        q = rng.choice(pool.questions)
        msgs = [{"role": "user", "content": f"Solve this problem and explain each step.\n\n{q}"}]
        out.append(req("burst", i, t, msgs, rng.randint(256, 512), in_spike=20 <= t < 30))
    return out


def spec(pool: TextPool, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for i, t in enumerate(poisson_times(rng, rate=0.5, duration=80)):
        passage = pool.passage(rng, rng.randint(300, 500))
        msgs = [{
            "role": "user",
            "content": "Rewrite the following passage word for word, only fixing obvious typos. "
                       f"Output the passage and nothing else.\n\n{passage}",
        }]
        out.append(req("spec", i, t, msgs, 640, ignore_eos=False))
    return out


WORKLOADS: dict[str, Callable[[TextPool, random.Random], list[dict[str, Any]]]] = {
    "chat": chat, "rag": rag, "agentic": agentic, "burst": burst, "spec": spec,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate deterministic traffic traces")
    ap.add_argument("--out", type=Path, default=Path("traces"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", choices=sorted(WORKLOADS))
    args = ap.parse_args()

    tok_path = find_snapshot_file(DEFAULT_TOKENIZER, "tokenizer.json")
    data_path = find_snapshot_file(DEFAULT_DATASET, "test.jsonl")
    tokenizer = Tokenizer.from_file(str(tok_path))
    pool = TextPool(data_path, tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.only or WORKLOADS:
        rng = random.Random(f"{args.seed}:{name}")
        reqs = WORKLOADS[name](pool, rng)
        path = args.out / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in reqs:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        prompt_toks = [sum(pool.ntok(m["content"]) for m in r["body"]["messages"]) for r in reqs]
        meta = {
            "workload": name,
            "seed": args.seed,
            "requests": len(reqs),
            "duration_s": round(reqs[-1]["t"], 1) if reqs else 0,
            "prompt_tokens_approx": {"min": min(prompt_toks), "max": max(prompt_toks), "total": sum(prompt_toks)},
            "max_tokens_total": sum(r["body"]["max_tokens"] for r in reqs),
            "tokenizer": DEFAULT_TOKENIZER,
            "dataset": f"{DEFAULT_DATASET} test.jsonl sha256:{sha256(data_path)}",
        }
        (args.out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{name:8s} {len(reqs):4d} requests  prompt≈{meta['prompt_tokens_approx']['total']:>8,} tok  "
              f"max out {meta['max_tokens_total']:>7,} tok  over {meta['duration_s']} s")


if __name__ == "__main__":
    main()
