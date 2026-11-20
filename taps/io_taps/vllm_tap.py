"""vLLM stat-logger plugin: one datagram per engine step.

Registered through the `vllm.stat_logger_plugins` entry point, so `vllm serve` loads it
without code changes. vLLM calls record() synchronously on the API server's output loop,
so this does only cheap attribute reads and one non-blocking sendto.

Per-step mix of prefill vs decode work needs `--enable-logging-iteration-details`.
Field names follow vLLM 0.34 (`vllm/v1/metrics/stats.py`); getattr keeps it tolerant of drift.
"""

from __future__ import annotations

from typing import Any

import vllm
from vllm.v1.metrics.loggers import StatLoggerBase

from . import UdpSender


def _g(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


class FlowTap(StatLoggerBase):
    def __init__(self, vllm_config: Any, engine_index: int = 0):
        self.engine_index = engine_index
        self.tx = UdpSender("vllm-tap")
        self.seq = 0
        sc = _g(vllm_config, "scheduler_config")
        cc = _g(vllm_config, "cache_config")
        pc = _g(vllm_config, "parallel_config")
        mc = _g(vllm_config, "model_config")
        spec = _g(vllm_config, "speculative_config")
        self.config = {
            "model": _g(mc, "model"),
            "max_model_len": _g(mc, "max_model_len"),
            "dtype": str(_g(mc, "dtype")),
            "max_num_seqs": _g(sc, "max_num_seqs"),
            "max_num_batched_tokens": _g(sc, "max_num_batched_tokens"),
            "chunked_prefill": _g(sc, "enable_chunked_prefill"),
            "scheduling_policy": str(_g(sc, "policy")),
            "async_scheduling": _g(sc, "async_scheduling"),
            "block_size": _g(cc, "block_size"),
            "gpu_memory_utilization": _g(cc, "gpu_memory_utilization"),
            "num_gpu_blocks": _g(cc, "num_gpu_blocks"),
            "prefix_caching": _g(cc, "enable_prefix_caching"),
            "kv_cache_dtype": str(_g(cc, "cache_dtype")),
            "tp": _g(pc, "tensor_parallel_size"),
            "pp": _g(pc, "pipeline_parallel_size"),
            "dp": _g(pc, "data_parallel_size"),
            "ep": _g(pc, "enable_expert_parallel"),
            "api_server_count": _g(pc, "_api_process_count"),
            "spec_method": _g(spec, "method"),
            "spec_tokens": _g(spec, "num_speculative_tokens"),
        }
        self.dp = int(self.config.get("dp") or 1)
        hello = {"type": "tap.hello", "engine": "vllm", "engine_version": vllm.__version__,
                 "engine_index": engine_index, "config": self.config, "url": self.tx.url}
        if self.dp > 1:
            hello["instance"] = f"{self.tx.instance}.dp{engine_index}"
        self.tx.send(hello)

    def log_engine_initialized(self) -> None:
        # Block count is only known after KV cache allocation.
        self.tx.send({"type": "tap.ready", "engine": "vllm", "engine_index": self.engine_index,
                      "config": self.config, "tap_sent": self.tx.sent, "tap_dropped": self.tx.dropped})

    def record(self, scheduler_stats: Any, iteration_stats: Any, mm_cache_stats: Any = None,
               engine_idx: int = 0) -> None:
        msg: dict[str, Any] = {"type": "engine.step", "engine": "vllm", "engine_idx": engine_idx, "seq": self.seq}
        if self.dp > 1:  # data-parallel ranks share a process tree; report each as its own instance
            msg["instance"] = f"{self.tx.instance}.dp{engine_idx}"
        self.seq += 1
        s = scheduler_stats
        if s is not None:
            msg["running"] = s.num_running_reqs
            msg["waiting"] = s.num_waiting_reqs
            msg["skipped_waiting"] = _g(s, "num_skipped_waiting_reqs", 0)
            msg["kv_usage"] = s.kv_cache_usage
            d = _g(s, "iteration_details")
            if d is not None:
                msg["iter"] = {
                    "i": d.iteration_index,
                    "ctx_reqs": d.num_ctx_requests,
                    "ctx_tokens": d.num_ctx_tokens,
                    "gen_reqs": d.num_generation_requests,
                    "gen_tokens": d.num_generation_tokens,
                    "elapsed_ms": d.elapsed_ms,
                    "dummy": d.is_dummy,
                }
            p = _g(s, "prefix_cache_stats")
            if p is not None and _g(p, "queries"):
                msg["prefix"] = {"requests": _g(p, "requests", 0), "queries": p.queries, "hits": _g(p, "hits", 0)}
            sd = _g(s, "spec_decoding_stats")
            if sd is not None:
                msg["spec"] = {
                    "drafts": _g(sd, "num_drafts", 0),
                    "draft_tokens": _g(sd, "num_draft_tokens", 0),
                    "accepted_tokens": _g(sd, "num_accepted_tokens", 0),
                    "accepted_per_pos": list(_g(sd, "num_accepted_tokens_per_pos", []) or []),
                }
            ev = _g(s, "kv_cache_eviction_events")
            if ev:
                msg["evictions"] = len(ev)
            kc = _g(s, "kv_connector_stats")
            if kc:
                msg["kv_connector"] = kc
        it = iteration_stats
        if it is not None:
            msg["gen_tokens"] = it.num_generation_tokens
            msg["preempted"] = it.num_preempted_reqs
            pts = _g(it, "prompt_token_stats")
            if pts is not None and _g(pts, "total"):
                msg["prompt"] = {
                    "computed": _g(pts, "computed", 0),
                    "cached": _g(pts, "local_cache_hit", 0),
                    "external": _g(pts, "external_kv_transfer", 0),
                    "total": _g(pts, "total", 0),
                }
            fin = _g(it, "finished_requests")
            if fin:
                msg["finished"] = [
                    {
                        "request_id": f.request_id,
                        "finish_reason": str(f.finish_reason),
                        "queued_s": f.queued_time,
                        "prefill_s": f.prefill_time,
                        "decode_s": f.decode_time,
                        "inference_s": f.inference_time,
                        "e2e_s": f.e2e_latency,
                        "prompt_tokens": f.num_prompt_tokens,
                        "cached_tokens": _g(f, "num_cached_tokens", 0),
                        "gen_tokens": f.num_generation_tokens,
                        "preemptions": f.num_preemptions,
                    }
                    for f in fin
                ]
        self.tx.send(msg)
