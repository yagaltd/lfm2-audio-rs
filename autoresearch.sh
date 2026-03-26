#!/bin/bash
set -euo pipefail

PORT="${PORT:-18080}"
MODEL_DIR="${LFM2_MODEL_DIR:-/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX}"
SERVER_BIN="./target/release/lfm2-server"
SERVER_LOG="$(mktemp /tmp/lfm2-realtime-server-XXXXXX.log)"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$SERVER_LOG"
}
trap cleanup EXIT

cargo build --quiet --release --features server --bin lfm2-server

RUST_LOG=warn "$SERVER_BIN" --model "$MODEL_DIR" --bind "127.0.0.1:${PORT}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

python - <<PY
import socket, time
port = ${PORT}
deadline = time.time() + 60
while time.time() < deadline:
    s = socket.socket()
    s.settimeout(0.2)
    try:
        s.connect(('127.0.0.1', port))
        s.close()
        raise SystemExit(0)
    except OSError:
        s.close()
        time.sleep(0.2)
raise SystemExit(1)
PY

BENCH_TIMEOUT_MS="${BENCH_TIMEOUT_MS:-180000}" \
BENCH_WS_URL="ws://127.0.0.1:${PORT}/ws/interleaved" \
node scripts/bench_realtime_ws.mjs
