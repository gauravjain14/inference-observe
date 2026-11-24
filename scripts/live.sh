#!/usr/bin/env bash
# The whole stack on one machine: SGLang with the tap on :18100, proxy :18000, collector :18220, advisor;
# then traffic, and optionally the push to a public relay.
#   scripts/live.sh up        start the engine, proxy, collector and advisor, and wait for the engine
#   scripts/live.sh copilot   replay GitHub Copilot agent sessions through the proxy (harness/copilot.py)
#   scripts/live.sh load      replay $TRACE through the proxy (harness/replay.py)
#   scripts/live.sh push      push the stream to the public relay at $PUSH_URL (harness/push.py)
#   scripts/live.sh status    show what is running
#   scripts/live.sh down      stop everything this script started
# Settings come from the environment or scripts/live.env; see scripts/live.env.example.
# Board: cd board && npm run dev, then http://127.0.0.1:18300/?ws=ws://127.0.0.1:18220/ws
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f scripts/live.env ]; then . scripts/live.env; fi
SGLANG_PY="${SGLANG_PY:-$ROOT/.venvs/sglang/bin/python}"
PY="${HARNESS_PY:-$ROOT/.venvs/harness/bin/python}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
CTX="${CTX:-32768}"
STATE="$ROOT/runs/local/.live"
mkdir -p "$STATE"
NAMES="push copilot load advisor proxy collector engine"

start() { # name log cmd...
  local name=$1 log=$2; shift 2
  setsid nohup "$@" > "$log" 2>&1 < /dev/null &
  echo $! > "$STATE/$name.pid"
}

run_dir() {
  [ -f "$STATE/run" ] || { echo "no run yet: scripts/live.sh up"; exit 1; }
  cat "$STATE/run"
}

case "${1:-}" in
  up)
    RUN="$ROOT/runs/local/live-$(date -u +%Y%m%d-%H%M%S)"
    mkdir -p "$RUN"
    echo "$RUN" > "$STATE/run"
    # shellcheck disable=SC2086  # ENGINE_ARGS holds several flags
    start engine "$RUN/engine.log" env -u LD_LIBRARY_PATH ${HF_HOME:+HF_HOME=$HF_HOME} CUDA_VISIBLE_DEVICES="${GPU:-0}" \
      DO_NOT_TRACK=1 PYTHONPATH="$ROOT/taps" SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 IO_TAP_PORT=18201 \
      IO_TAP_URL=http://127.0.0.1:18100 IO_TAP_INSTANCE=engine-0 \
      "$SGLANG_PY" -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 18100 --tp-size 1 \
      --enable-metrics --enable-cache-report --context-length "$CTX" ${ENGINE_ARGS:-}
    start collector "$RUN/collector.log" "$PY" -m harness.collector --engine sglang --model "$MODEL" --gpus "${GPU:-0}" \
      --out "$RUN" --port 18220 --tap-port 18201 --notes "$RUN/notes.jsonl" ${LABEL:+--label "$LABEL"}
    start proxy "$RUN/proxy.log" "$PY" -m harness.proxy --upstream http://127.0.0.1:18100 --events "$RUN/proxy.jsonl" \
      --rid-in-body
    for _ in $(seq 1 120); do
      curl -sf -o /dev/null http://127.0.0.1:18100/health && break
      if ! kill -0 "$(cat "$STATE/engine.pid")" 2>/dev/null; then echo "engine exited:"; tail -20 "$RUN/engine.log"; exit 1; fi
      sleep 3
    done
    curl -sf -o /dev/null http://127.0.0.1:18100/health || { echo "engine not healthy after 6 min"; exit 1; }
    start advisor "$RUN/advisor.log" "$PY" -m harness.advisor --collector ws://127.0.0.1:18220/ws \
      --post http://127.0.0.1:18220/note
    echo "up: $RUN"
    ;;
  copilot)
    shift
    RUN="$(run_dir)"
    # shellcheck disable=SC2086  # COPILOT_ARGS holds several flags
    if [ $# -eq 0 ]; then set -- "${COPILOT_TRACE:?set COPILOT_TRACE or pass the shards folder}" ${COPILOT_ARGS:-}; fi
    start copilot "$RUN/copilot.out" "$PY" -m harness.copilot "$@" --base http://127.0.0.1:18000 --out "$RUN"
    echo "copilot replay: $RUN/copilot.log"
    ;;
  load)
    RUN="$(run_dir)"
    start load "$RUN/load.out" "$PY" -m harness.replay "${TRACE:-$ROOT/traces/demo.jsonl}" \
      --base http://127.0.0.1:18000 --out "$RUN" --no-salt
    echo "replay: $RUN/load.out"
    ;;
  push)
    RUN="$(run_dir)"
    start push "$RUN/push.log" "$PY" -m harness.push --relay "${2:-${PUSH_URL:?set PUSH_URL, e.g. wss://live.<domain>/ingest}}"
    echo "push: $RUN/push.log"
    ;;
  status)
    for n in $NAMES; do
      f="$STATE/$n.pid"
      if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then echo "$n: running"; else echo "$n: stopped"; fi
    done
    ;;
  down)
    for n in $NAMES; do
      f="$STATE/$n.pid"
      [ -f "$f" ] || continue
      pid=$(cat "$f")
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
      rm -f "$f"
    done
    echo "down"
    ;;
  *) echo "usage: $0 up|copilot|load|push|status|down"; exit 2 ;;
esac
