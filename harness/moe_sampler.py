"""MoE expert-routing sampler for SGLang: per-expert load and per-layer imbalance, every few seconds.

Uses SGLang's own expert-distribution recorder (launch with
`--expert-distribution-recorder-mode stat`). Each cycle asks the server to dump the counts it
accumulated since the last dump (the dump also resets them), loads the file, and sends one
`moe.tick` datagram to the collector:

  {type: "moe.tick", layers, experts, window_s, tokens,
   load: [share of routed tokens per expert, summed over layers],
   layer_imbalance: [max/mean routed tokens per expert, per layer],
   imbalance_max, hot: [top experts],
   grid_b64: layers x experts bytes, row-major, each = round(load / layer mean * grid_scale), capped at 255}

Runs in the SGLang venv (needs torch to read the dump).

  python -m harness.moe_sampler --engine-url http://127.0.0.1:18100 --dir <run>/moe
"""

from __future__ import annotations

import argparse
import base64
import time
import urllib.request
from pathlib import Path

import torch

from io_taps import UdpSender

GRID_SCALE = 32  # grid byte = relative load x 32: 32 = fair share, 255 = ~8x


def call(url: str) -> None:
    urllib.request.urlopen(urllib.request.Request(url, data=b"", method="POST"), timeout=30).read()


def latest(dir_: Path, after: float) -> Path | None:
    files = [p for p in dir_.glob("expert_distribution_recorder_*.pt") if p.stat().st_mtime >= after]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def summarize(counts: torch.Tensor) -> dict:
    """counts: [buffered_steps, layers, experts] or [layers, experts] -> load shares and imbalance."""
    if counts.dim() == 3:
        counts = counts.sum(dim=0)
    counts = counts.double()
    total = counts.sum().item()
    per_expert = counts.sum(dim=0)
    load = (per_expert / per_expert.sum()).tolist() if total > 0 else [0.0] * counts.shape[1]
    mean = counts.mean(dim=1).clamp(min=1e-9)
    imb = (counts.max(dim=1).values / mean)
    active = counts.sum(dim=1) > 0
    imb_list = [round(x, 3) if a else None for x, a in zip(imb.tolist(), active.tolist())]
    hot = torch.topk(per_expert, k=min(8, per_expert.numel())).indices.tolist() if total > 0 else []
    # Full layer x expert grid for the heatmap: each cell's load relative to its layer's mean
    # (1.0 = fair share), as one byte per cell: round(rel * GRID_SCALE), capped at 255 (~8x).
    rel = (counts / mean.unsqueeze(1)).clamp(max=255 / GRID_SCALE)
    grid = (rel * GRID_SCALE).round().to(torch.uint8).flatten().numpy().tobytes()
    vals = [x for x in imb_list if x is not None]
    return {
        "layers": int(counts.shape[0]), "experts": int(counts.shape[1]), "tokens": int(total),
        "load": [round(x, 5) for x in load], "layer_imbalance": imb_list,
        "imbalance_max": round(max(vals), 3) if vals else None,
        "imbalance_mean": round(sum(vals) / len(vals), 3) if vals else None,
        "hot": hot,
        "grid_b64": base64.b64encode(grid).decode(), "grid_scale": GRID_SCALE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample SGLang expert routing into moe.tick events")
    ap.add_argument("--engine-url", default="http://127.0.0.1:18100")
    ap.add_argument("--dir", type=Path, required=True, help="SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR of the engine")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--model", default=None, help="HF model id, to read num_experts_per_tok from its config")
    a = ap.parse_args()
    top_k = None
    if a.model:
        try:
            import json as _json

            from huggingface_hub import try_to_load_from_cache
            cfg = try_to_load_from_cache(a.model, "config.json")
            if isinstance(cfg, str):
                c = _json.load(open(cfg))
                top_k = c.get("num_experts_per_tok") or (c.get("text_config") or {}).get("num_experts_per_tok")
        except Exception as e:
            print("could not read top_k:", e, flush=True)
    tx = UdpSender("moe-sampler")
    base = a.engine_url.rstrip("/")
    a.dir.mkdir(parents=True, exist_ok=True)

    while True:  # wait for the engine
        try:
            call(base + "/start_expert_distribution_record")
            break
        except Exception:
            time.sleep(3)
    print("recording expert distribution", flush=True)

    last = time.time()
    while True:
        time.sleep(a.every)
        t0 = time.time()
        try:
            call(base + "/dump_expert_distribution_record")
        except Exception as e:
            print("dump failed:", e, flush=True)
            time.sleep(3)
            try:
                call(base + "/start_expert_distribution_record")
            except Exception:
                pass
            continue
        path = None
        for _ in range(20):
            path = latest(a.dir, t0 - 0.5)
            if path:
                break
            time.sleep(0.1)
        if not path:
            continue
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            msg = summarize(data["logical_count"])
            msg.update(type="moe.tick", engine="sglang", window_s=round(t0 - last, 2), top_k=top_k)
            if msg["tokens"] > 0:
                tx.send(msg)
        except Exception as e:
            print("parse failed:", e, flush=True)
        finally:
            last = t0
            for p in a.dir.glob("expert_distribution_recorder_*.pt"):
                p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
