# Event format

Everything the board draws arrives as JSON events on one WebSocket (`ws://<collector>:18200/ws`).
The collector also writes them, one per line, to `<run>/collector.jsonl`; that file replays as is.

Every event has `type` and `t_ns` (Unix time in nanoseconds). Frames may carry one event or an array.
A field that is missing or `null` means "not reported"; the board shows it as unknown.

## Sources

| Source | Process | Transport | Events |
|---|---|---|---|
| Tap | inside the engine (SGLang plugin or vLLM stat logger) | UDP to the collector, `:18201` | `engine.step`, `kv.map`, `tap.hello` |
| Proxy | `io-proxy`, in front of the engine | WebSocket `/_events`, read by the collector | `req.*` |
| Collector | `io-collector` | adds its own | `hello`, `engine.tick`, `gpu.tick` |
| Expert sampler | `harness/moe_sampler.py` (SGLang) | UDP to the collector | `moe.tick` |
| Advisors, tools | anything | `POST /note` to the collector | `advisor.note` |

## `hello`

Sent first on every connection, and again when coverage changes.

| Field | Meaning |
|---|---|
| `engine`, `engine_version`, `model` | what is being observed |
| `gpus`, `gpu_ids` | GPUs sampled by the collector (on its own machine) |
| `coverage` | `{requests, steps, kv, spec, gpu, moe}`: which signals are live. The board leaves the rest unstained. |
| `config` | engine settings reported by the tap (page size, chunked prefill, TP/DP/PP, nodes, reporting rank, …) |
| `instances` | one entry per engine instance (DP rank, replica, prefill or decode worker): `{role, url}` |
| `label`, `replay` | set for recordings: `label: "Recorded specimen"`, `replay: {file, speed, recorded_utc}` |

## Requests (`req.*`, from the proxy)

`rid` is the proxy's request id (also sent to the engine, so tap data joins on it).

| Event | Fields |
|---|---|
| `req.arrive` | `endpoint, model, stream, max_tokens, workload` (from the `X-Workload` header), `prompt_chars, n_messages`, `prefix_family, prefix_chars` (hash and length of the prompt's first characters) |
| `req.admit` | `inflight, waiting` (proxy admission; `--max-inflight` caps it) |
| `req.route` | `worker` (replica routing) |
| `req.prefill_start`, `req.prefill_done`, `req.decode_start` | `worker`, `prefill_ms` (prefill/decode disaggregation) |
| `req.prefill_stats` | `prompt_tokens, cached_tokens` (engine-reported, from the stream's usage) |
| `req.first_token` | `ttft_ms` |
| `req.chunk` | `i, n` (streamed chunks, batched) |
| `req.done` | `e2e_ms, n_chunks, finish_reason, usage` |
| `req.cancel`, `req.error` | `stage`, `status`, `detail` |

## `engine.step` (from the tap, one per scheduler step)

| Field | Meaning |
|---|---|
| `instance` | `IO_TAP_INSTANCE`, plus `.dp<N>` per data-parallel rank |
| `running, waiting, kv_usage` | scheduler state after the step; `kv_usage` counts tokens held by running requests |
| `iter` | `{ctx_reqs, ctx_tokens, gen_reqs, gen_tokens, elapsed_ms}`: prefill and decode work in this step |
| `batch` | `{reqs: [[rid, "p" or "d", tokens], …]}`: who was in the step (SGLang) |
| `prefix` | `{queries, hits}` prefix-cache lookups in this step |
| `spec` | speculative decoding drafts and accepted tokens |
| `finished` | per finished request: `queued_s, prefill_s, decode_s, prompt_tokens, cached_tokens, gen_tokens, preemptions, finish_reason` |

## `kv.map` (SGLang tap, once a second)

The KV pool as `bins` equal cells of `bin_tokens` tokens each.

| Field | Meaning |
|---|---|
| `capacity_tokens, used_tokens, owned_tokens` | pool size; allocated tokens (live and cached); tokens held by running requests |
| `owners` | per running request: `{rid, tokens, prefix_tokens, prefix_ranges, own_ranges}` (cell ranges) |
| `shared` | cell ranges of prefixes held by two or more running requests |
| `cached` | cell ranges of prefixes kept for reuse with no running owner |
| `free` | free cell ranges |

## `moe.tick` (SGLang expert sampler, every 2 s)

`layers, experts, top_k, window_s, tokens`; `load` (share of routed tokens per expert, all layers);
`layer_imbalance` (busiest expert ÷ average expert, per layer); `hot` (busiest experts);
`grid_b64` (layers × experts bytes: load ÷ layer average × `grid_scale`, capped at 255).

## `engine.tick`, `gpu.tick` (collector, 10 times a second)

`engine.tick`: `running, waiting, kv_usage, prefix_hit_rate, prompt_tok_s, gen_tok_s, steps_per_s`, and
the same per instance under `instances`. `source` says whether it came from the tap or from the
engine's Prometheus `/metrics`.

`gpu.tick`: per GPU `{i, util, mem_util, mem_used_gb, mem_total_gb, power_w, sm_clock_mhz}` (NVML).

## `advisor.note` (`POST /note`)

```json
{"id": "kv-pressure-1", "agent": "advisor", "title": "…", "body": "…",
 "predicted": "…", "measured": "…", "region": "kv", "status": "proposed"}
```

Posting the same `id` again updates the note. `status`: `info`, `proposed`, `testing`, `confirmed`,
`rejected`, `neutral`, `failed`. `region` (`membrane`, `queue`, `kv`, `decode`, `compute`) places the
note's marker on the plate.
