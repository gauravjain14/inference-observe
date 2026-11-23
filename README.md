# inference-observe

A live view inside an LLM inference engine. Every request, scheduler step, KV-cache block and expert
hit from **SGLang** or **vLLM** is drawn as it happens, and advisor agents post tuning proposals next
to the evidence.

![The board, SGLang serving GitHub Copilot agent sessions](docs/board.png)

Video (31 s): [docs/demo.mp4](docs/demo.mp4). SGLang 0.5.19 on one RTX 4060 Ti, serving GitHub Copilot
coding-agent sessions replayed from the June 2026 production traces. A new session's 11,139-token prompt
takes 2.07 s to its first token; the same session's next call finds 10,973 of its 11,006 tokens in the
cache and takes 100 ms.

## What the board shows

| On the board | Signal | Source |
|---|---|---|
| Crimson cells on the left | requests waiting at the proxy, then admitted into the engine | proxy |
| Blue colonies inside the body | one per running request: its KV cache, to scale against the KV pool. The dark centre is the prefix it found in the cache | tap |
| Dark core with a ring | KV that two or more running requests hold at once: one copy in memory, several readers | tap |
| Hatched field | KV kept in the cache with no running request (reusable until evicted) | tap |
| Memory map (press **M**) | the KV pool in its real order, each cell split by its tokens: own, reused from the cache, shared, cached, free | tap (SGLang) |
| Crimson veins | one per running request, from its colony, through its traffic class and engine instance; width = tokens/s | proxy |
| Green neurons | GPUs; they fire on each scheduler step | NVML, tap |
| Fig. 3 | every scheduler step: prefill tokens up, decoding requests down | tap |
| Fig. 4 | dense models: tokens reused from cache vs computed, and time to first token, per request. MoE models: load on the busiest expert of each layer | proxy, tap |
| Fig. 2 | prompt vs output tokens per request (prefill-heavy vs decode-heavy) | proxy |
| Field notes | advisor proposals: evidence, a config change, the predicted effect | advisors |

Every number carries a mark: ● measured by the engine, ◐ derived from measured values, ○ estimated.
Signals the engine does not report stay blank. Press **?** on the board for the key, **F** to follow
one request through the engine, scroll to zoom, click anything to inspect it.

## Quick start: replay a recording

Needs Node 20 or later.

```bash
cd board
npm install
npm run dev
```

Open <http://127.0.0.1:18300/?replay=/samples/sglang-copilot-agent.jsonl&loop=1>.

The sample is two minutes of SGLang 0.5.19 serving Qwen2.5-Coder-3B-Instruct on one RTX 4060 Ti, with
GitHub Copilot agent sessions replayed from the production traces (see
[Agent sessions](#agent-sessions-github-copilot-traces)).

## Connect SGLang

Tested with SGLang 0.5.19 and 0.5.23.

1. Install the tap into SGLang's Python environment. SGLang loads it through its plugin entry point.

   ```bash
   pip install -e taps
   ```

2. Start SGLang with metrics and cache reporting. `IO_TAP_URL` is this server's address; the collector
   uses it to match engine data with proxy traffic.

   ```bash
   IO_TAP_URL=http://127.0.0.1:30000 python -m sglang.launch_server \
     --model-path Qwen/Qwen3-4B-Instruct-2507 --port 30000 \
     --enable-metrics --enable-cache-report
   ```

3. Install the harness in its own environment (Python 3.12) and start the proxy and the collector.

   ```bash
   uv venv && source .venv/bin/activate && uv pip install -e .
   io-proxy --upstream http://127.0.0.1:30000 --rid-in-body &
   io-collector --engine sglang --engine-url http://127.0.0.1:30000 --gpus 0 --out runs/live &
   ```

4. Start the board and open <http://127.0.0.1:18300/>.

   ```bash
   cd board && npm install && npm run dev
   ```

5. Send requests to the proxy: `http://127.0.0.1:18000/v1/chat/completions`. An `X-Workload: <name>`
   header gives a request a traffic class on the board.

The board's header shows `SGLang <version> · <model>` and `live`. The collector records everything to
`runs/live/collector.jsonl`.

**Expert routing (MoE models).** Start SGLang with its expert recorder, then run the sampler with
SGLang's Python from the repository root. It works with models whose SGLang implementation records
layer indices (Qwen-MoE, DeepSeek).

```bash
SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/tmp/io-moe IO_TAP_URL=http://127.0.0.1:30000 \
  python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --port 30000 \
  --enable-metrics --enable-cache-report --expert-distribution-recorder-mode stat
python -m harness.moe_sampler --engine-url http://127.0.0.1:30000 --dir /tmp/io-moe \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507
```

## Connect vLLM

Tested with vLLM 0.32 and 0.34.

1. Install the tap into vLLM's Python environment. vLLM loads it as a stat logger.

   ```bash
   pip install -e taps
   ```

2. Start vLLM with per-step details and cached-token reporting.

   ```bash
   IO_TAP_URL=http://127.0.0.1:8000 vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
     --enable-logging-iteration-details --enable-prompt-tokens-details
   ```

3. Start the proxy and the collector.

   ```bash
   io-proxy --upstream http://127.0.0.1:8000 &
   io-collector --engine vllm --engine-url http://127.0.0.1:8000 --gpus 0 --out runs/live &
   ```

4. Start the board and send requests to the proxy, as for SGLang.

### What each tap reports

| | SGLang | vLLM |
|---|---|---|
| Per step: prefill and decode work, running, waiting, KV usage | yes | yes |
| Per step: which requests were in the batch | yes | no |
| Per request: queue, prefill and decode time, cached tokens | yes | yes |
| Prefix-cache lookups and hits | yes | yes |
| Speculative decoding drafts and acceptance | yes | yes |
| KV memory map, every slot classified | yes | no |
| Expert routing | yes, with the sampler | no |

## Multi-GPU and multi-node: connect to rank 0

The tap loads in every engine process, but only one process per engine instance sends data:

- **SGLang**: the reporting rank (pipeline rank 0, attention tensor-parallel rank 0, attention
  context-parallel rank 0), the same rank SGLang uses for its own metrics. It runs on node 0. With
  data parallelism, each DP rank reports as a separate instance, `<instance>.dp<N>`.
- **vLLM**: the tap is a stat logger in the API server, which runs on the head node and receives the
  stats of every engine. Data-parallel engines report as `<instance>.dp<N>`.

The collector only needs to reach node 0. Example: SGLang on two nodes of 8 GPUs, tensor parallel 16.

1. On both nodes, install the tap into SGLang's environment: `pip install -e taps`. Ranks other than
   the reporting rank load it and send nothing.

2. Start node 0.

   ```bash
   IO_TAP_URL=http://<node0>:30000 python -m sglang.launch_server --model-path <model> \
     --tp 16 --nnodes 2 --node-rank 0 --dist-init-addr <node0>:5000 \
     --host 0.0.0.0 --port 30000 --enable-metrics --enable-cache-report
   ```

3. Start node 1 with the same command and `--node-rank 1`.

4. On node 0, start the proxy and the collector against node 0's server.

   ```bash
   io-proxy --upstream http://127.0.0.1:30000 --rid-in-body &
   io-collector --engine sglang --engine-url http://127.0.0.1:30000 --gpus 0,1,2,3,4,5,6,7 --out runs/live &
   ```

5. On your machine, forward the two ports, start the board and send requests to `http://127.0.0.1:18000`.

   ```bash
   ssh -N -L 18200:127.0.0.1:18200 -L 18000:127.0.0.1:18000 <node0>
   ```

- To run the collector on another machine, set `IO_TAP_HOST=<collector host>` on node 0 (default
  `127.0.0.1`).
- With data parallelism across nodes, DP ranks on other nodes report too: set `IO_TAP_HOST` on every node.
- GPU telemetry covers the GPUs of the machine that runs the collector.
- vLLM across nodes (Ray, or data parallel with `--headless` workers): install the tap on the head node
  and start the proxy and the collector there, against the API server.

Tested on one node: vLLM with tensor parallel 2, data parallel 2, two replicas, and prefill/decode
disaggregation over NIXL; SGLang on one GPU. Not yet run across nodes.

## Advisors

```bash
io-advisor --collector ws://127.0.0.1:18200/ws --post http://127.0.0.1:18200/note
```

The advisor watches the stream and posts proposals under Field notes, each with its evidence, a config
change for the running engine and the predicted effect. Examples: requests queue because the KV cache
is full; shared prefixes are recomputed. Any tool can post notes: see [docs/events.md](docs/events.md).

## Workloads

`io-copilot` replays agent sessions in a closed loop. `io-replay` sends other traces through the proxy at
their scheduled times, so overload shows up as queueing.

### Agent sessions (GitHub Copilot traces)

The GitHub Copilot coding-agent traces (Azure Public Dataset, June 2026) record 301k agent sessions:
every model call with its timestamp, duration, and prompt, completion and cached token counts, and every
tool run with its duration. They contain no text. `io-copilot` replays them against a live engine.

- **Closed loop.** Each session starts at its recorded time. Each call waits for the call it followed,
  then for the recorded gap (a tool run or a user pause). The engine's own speed sets the pace.
- **Prompts.** Each call sends the session's conversation so far (the previous prompt and the engine's
  own answer) plus filler text for the new content (tool output, the user's next message), sized to the
  recorded prompt. A prompt much shorter than the conversation (context compaction) keeps only the
  system part.
- **Answers** are generated to the recorded length.
- **Cache.** The engine's prefix cache decides what is reused. Production's cached count is logged next
  to the engine's. The trace has no text, so prompts shared across sessions are not reproduced.

Traffic classes on the board: new session, user turn, agent loop, and session in progress (the first
call of a session that began before the replayed window; its cache starts empty).

Run it on one GPU:

```bash
curl -LO https://github.com/Azure/AzurePublicDataset/releases/download/ghcp-coding-agent-2026/date.2026-06-06.tar.gz
tar -xzf date.2026-06-06.tar.gz          # creates date=2026-06-06/

IO_TAP_URL=http://127.0.0.1:30000 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Coder-3B-Instruct --port 30000 \
  --enable-metrics --enable-cache-report --context-length 32768 --cuda-graph-max-bs-decode 32
io-proxy --upstream http://127.0.0.1:30000 --rid-in-body &
io-collector --engine sglang --engine-url http://127.0.0.1:30000 --gpus 0 --out runs/live \
  --label "GitHub Copilot agent traces, June 2026" &
io-copilot 'date=2026-06-06' --start 2026-06-06T13:00 --duration 14400 \
  --sample 0.12 --prompt-scale 0.125 --out runs/live --loop
```

- `--sample` replays that fraction of sessions, and `--prompt-scale` scales prompt sizes. These values
  fit a 16 GB GPU, with a 166k-token KV cache.
- `--cuda-graph-max-bs-decode 32`: on a small GPU, SGLang sizes CUDA graphs for decode batches up to 8,
  and this replay runs more requests than that at once.
- Every 30 s, `runs/live/copilot.log` gets a line with the share of prompt tokens the engine served
  from its cache, next to production's, per traffic class. When the replay stops it writes
  `copilot-summary.json` and `copilot-calls.jsonl`.

On the run behind the sample and the video (349 calls):

| Call | Prompt tokens from cache here | In production | Time to first token, p50 / p90 |
|---|---|---|---|
| agent loop (305) | 98% | 95% | 95 ms / 210 ms |
| user turn (15) | 80% | 87% | 138 ms / 3.6 s |
| new session (11) | 0% | 5% | 424 ms / 2.1 s |

### Production traffic (Mooncake)

Mooncake's open traces (Kimi, FAST '25) give, per request, the arrival
time, the input and output lengths and one hash ID per 512-token prompt block. `io-mooncake` turns each
hash ID into a fixed block of text, so prompts share prefixes exactly where the originals did.

```bash
curl -LO https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl
io-mooncake toolagent_trace.jsonl --out traces/mooncake-agent.jsonl \
  --block-tokens 256 --duration 70 --stretch 3.5 --max-input 12000 --max-output 256 --spread
io-replay traces/mooncake-agent.jsonl --base http://127.0.0.1:18000 --out runs/live --no-salt
```

`--block-tokens` and `--stretch` scale the trace to a smaller GPU. The trace records arrivals at
3-second resolution; `--spread` places each timestamp's requests evenly across those 3 seconds. Replay
Mooncake traces with `--no-salt`, which keeps the prompts as written.

### Synthetic workloads

`io-traces --out traces` builds chat, agentic, RAG, burst and speculative
workloads from MATH-500 prompts. By default `io-replay` makes each replay's prompts unique, so a
repeated replay starts with a cold cache.

## Checking the numbers

`board/scripts/audit-live.mjs` opens the board on a live collector and compares its numbers with
independent measurements: SGLang's own gauges for running requests and KV, and the token counts the
engine returned through the API for every request. Needs Playwright with Chrome.

```bash
cd board && node scripts/audit-live.mjs http://127.0.0.1:30000 ../runs/live/collector.jsonl 120
```

On the run behind the sample (350 finished requests):

| Board | Board value | Independent value |
|---|---|---|
| prompt tokens | 2,838,646 | 2,838,646 (API usage per request; 0 requests differ) |
| cached tokens | 2,765,128 | 2,765,128 (API usage per request) |
| generated tokens | 139,697 | 139,679 (API usage; requests still streaming) |
| KV maps where cached = SGLang's evictable tokens at the same instant | 790 | 790 |
| KV live / cached / free | 54.6% / 44.3% / 1.1% | 54.5% / 44.3% / 1.2% (`kv_used`, `kv_evictable`, `kv_available` gauges) |
| TTFT p50 / p90, last 60 s | 102.79 / 270.56 ms | 102.74 / 270.49 ms (proxy events) |

SGLang updates its gauges on its own logging interval, so under load they can trail the board by about a
second.

## Host it publicly

The board goes on GitHub Pages and the live stream through a relay on Cloudflare: a Worker with one Durable
Object, on the free plan. The engine's machine only connects outward: `io-push` sends the stream to the relay,
and viewers read it from there. The engine, the proxy and the collector stay on `127.0.0.1`.

```
browser ── board ─────────────────── GitHub Pages
   └──── wss://live.<domain>/ws ──── Cloudflare Worker relay ◄── io-push (outbound) ── collector :18220
```

**1. Relay on Cloudflare.** This needs a Cloudflare account with your domain on it.

```bash
cd relay-worker
npm install
cp wrangler.toml wrangler.local.toml   # set ALLOWED_ORIGINS and the routes line for live.<domain>
npx wrangler login
npx wrangler deploy -c wrangler.local.toml
mkdir -p ~/.config/inference-observe
(umask 077; python3 -c "import secrets; print(secrets.token_urlsafe(48))" > ~/.config/inference-observe/push-token)
tr -d '\n' < ~/.config/inference-observe/push-token | npx wrangler secret put INGEST_TOKEN -c wrangler.local.toml
```

If the deploy reports a missing `workers.dev` subdomain, open **Workers & Pages** in the Cloudflare dashboard
once, then deploy again.

The relay:
- serves `/ws` as a GET WebSocket only, to the page's origin;
- disconnects any viewer that sends data;
- accepts at most 200 viewers and 4 per address (IPv6 by /64), each for at most an hour (the board reconnects by
  itself);
- accepts `/ingest` only with the token, and answers every other request with 404;
- forwards text without parsing it. `io-push` sends the state a joining viewer needs every 5 seconds.

On the free plan:
- `io-push` sends 4 batches a second, about 17,000 requests a day of the 100,000 allowed;
- the relay stays active, using about 11,000 of the 13,000 GB-seconds allowed a day;
- if a limit is reached, the board goes dark until 00:00 UTC.

**2. Push from the engine's machine.** It reads the token from `~/.config/inference-observe/push-token`.

```bash
io-push --relay wss://live.<domain>/ingest
```

**3. Board on GitHub Pages.**

```bash
cd board
VITE_PUBLIC_WS=wss://live.<domain>/ws npx vite build --base=/<repo>/
```

A build with `VITE_PUBLIC_WS`:
- shows only that feed and recordings from its own site; `?ws=` and other sites' `?replay=` are ignored;
- carries a Content-Security-Policy that allows only its own files and that feed;
- plays the bundled recording when the feed is down.

Publish `board/dist/` on a `gh-pages` branch, then choose **Settings → Pages → Deploy from a branch**.

**4. Check it from outside.**

```bash
io-check-public wss://live.<domain>/ws --origin https://<user>.github.io --page https://<user>.github.io/<repo>/
```

This connects as the page, tries other origins, other paths (including the proxy's and the collector's), sending
data, and too many connections. It also reads the page's policy. Every line should read PASS.

**What the stream contains:** the engine's version and configuration, the model, GPU telemetry, per-request
timing and token counts, and a hash of each prompt's first 1,024 characters. It never contains prompt or
response text.

**Keep private:**
- the push token, `~/.config/inference-observe/push-token`;
- wrangler's login, which `npx wrangler logout` removes once you have deployed.

**Without Workers.** `io-relay` serves the same read-only stream from your machine, for a Cloudflare tunnel. Point
the tunnel's ingress at it with `path: ^/ws$` and `service: http://127.0.0.1:18240`, and answer everything else
with `http_status:404`. The Worker keeps your machine free of incoming connections.

PyTorch's distributed store inside SGLang listens on all network interfaces on a random port. Block inbound
connections with a host firewall, for example `ufw default deny incoming`.

## Recordings

- The collector writes `<out>/collector.jsonl`.
- Replay it in the browser: copy it to `board/public/samples/` and open `?replay=/samples/<file>.jsonl&loop=1`.
- Or serve it like a live collector: `io-replay-ws runs/live --loop`, then open the board as usual.

## Reference

| Port | |
|---|---|
| 18000 | proxy (OpenAI-compatible) |
| 18200 | collector: board stream `ws://…/ws`, `POST /note`, `GET /health` |
| 18201/udp | collector: tap datagrams (`IO_TAP_HOST`, `IO_TAP_PORT`) |
| 18300 | board |

Board URL parameters: `?ws=ws://<host>:<port>/ws`, `?replay=<url>&loop=1&speed=2`,
`?mock=steady|burst|pd|rep` (synthetic data, labelled as such on the board).

Collector: `--label` sets the caption on the board's plate.

Tap environment: `IO_TAP_URL` (this server's address), `IO_TAP_INSTANCE` (instance name, default
`engine`), `IO_TAP_HOST` and `IO_TAP_PORT` (collector), `IO_KV_MAP_EVERY` (seconds between KV maps,
default 1).

| Path | |
|---|---|
| `taps/` | engine plugins: SGLang (`sglang.srt.plugins`) and vLLM (`vllm.stat_logger_plugins`) |
| `harness/` | proxy, collector, advisor, agent-session replay, public push and relay, replay tools, trace converters, expert sampler |
| `relay-worker/` | the public relay on Cloudflare (Worker and Durable Object) |
| `board/` | the board (TypeScript, WebGL2, Vite); `scripts/evaluate.mjs` is its visual test suite |
| `docs/events.md` | the event format |

Board checks: `cd board && npx tsc --noEmit -p . && BOARD_URL=http://127.0.0.1:18300/ node scripts/evaluate.mjs`
(Playwright with Chrome).

## License and data

Apache-2.0.

- GitHub Copilot coding-agent traces: [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md),
  CC-BY. Liu, Qiu, Goiri, Fonseca, Bianchini, Choukse, "Agentic Coding in the Wild: Characterizing GitHub
  Copilot Traces at Production Scale", arXiv 2608.00101, 2026.
- Mooncake traces: [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake).
- Synthetic workloads: [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).
