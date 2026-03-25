#!/bin/bash
set -euo pipefail

# Configuration
MODEL_DIR="${LFM2_MODEL_DIR:-/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX}"
SERVER_BIN="${SERVER_BIN:-./target/release/lfm2-server}"
PORT="${PORT:-8080}"
TIMEOUT="${TIMEOUT:-120}"
PRECISION="${LFM2_PRECISION:-q4}"
LOG_FILE=$(mktemp /tmp/lfm2-benchmark-XXXXXX.log)

# Cleanup on exit
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$LOG_FILE" "${LOG_FILE}.filtered"
}
trap cleanup EXIT

# Check server binary exists
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: Server binary not found at $SERVER_BIN" >&2
    echo "Run: cargo build --release --features server" >&2
    exit 1
fi

# Start server with timing logs
echo "Starting server..." >&2
RUST_LOG=info,lfm2_server=info NO_COLOR=1 "$SERVER_BIN" --model "$MODEL_DIR" --bind "127.0.0.1:$PORT" --precision "$PRECISION" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..." >&2
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready on port $PORT" >&2
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "ERROR: Server failed to start within 30s" >&2
        cat "$LOG_FILE" >&2
        exit 1
    fi
    sleep 1
done

# Function to run a query and extract metrics
run_query() {
    local query="$1"
    local query_label="$2"
    local max_chunks="${3:-50}"

    echo "Running query: $query_label (capturing $max_chunks chunks)" >&2

    # Send WebSocket message and capture first N chunks
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    node -e "
        const WebSocket = require('$script_dir/node_modules/ws');
        const ws = new WebSocket('ws://localhost:$PORT/ws/interleaved');

        ws.on('open', () => {
            ws.send(JSON.stringify({type: 'user.text', text: '$query'}));
        });

        let chunkCount = 0;
        ws.on('message', (data, isBinary) => {
            // Count binary chunks (audio)
            if (isBinary) {
                chunkCount++;
                if (chunkCount >= $max_chunks) {
                    console.log('Captured ' + chunkCount + ' audio chunks');
                    ws.close();
                    setTimeout(() => process.exit(0), 100);
                }
                return;
            }
            
            try {
                const msg = JSON.parse(data.toString());
                if (msg.type === 'assistant.turn') {
                    console.log('Query completed with ' + chunkCount + ' audio chunks');
                    ws.close();
                    setTimeout(() => process.exit(0), 100);
                }
            } catch (e) {
                // Ignore parse errors for non-JSON messages
            }
        });

        // Timeout after ${TIMEOUT}s
        setTimeout(() => {
            console.log('Captured ' + chunkCount + ' chunks before timeout');
            ws.close();
            setTimeout(() => process.exit(0), 100);
        }, ${TIMEOUT}000);
    " 2>&1

    echo "Query '$query_label' completed" >&2
}

# Run test queries - capture first 50 frames each to measure timing
echo "" >&2
echo "=== Running benchmark queries ===" >&2
run_query "Say hi." "short_hi" 50
sleep 2  # Brief pause between queries  
run_query "Tell me a short joke." "short_joke" 50

# Give server a moment to flush logs
sleep 2

# Stop server to ensure all logs are flushed
echo "Stopping server..." >&2
if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
fi
unset SERVER_PID

# Parse logs and extract metrics
echo "" >&2
echo "=== Parsing timing metrics ===" >&2

# Extract frame_gap_ms and decode_elapsed_ms from logs
# Log format: frame_gap_ms=87 decode_elapsed_ms=24

# Output structured metrics for autoresearch
node -e "
    const fs = require('fs');
    const log = fs.readFileSync('$LOG_FILE', 'utf8');

    // Extract all frame_gap_ms values
    const frameGapMatches = log.matchAll(/frame_gap_ms=(\d+)/g);
    const frameGaps = [...frameGapMatches].map(m => parseInt(m[1])).filter(v => v > 0);

    // Extract all decode_elapsed_ms values
    const decodeMatches = log.matchAll(/decode_elapsed_ms=(\d+)/g);
    const decodeTimes = [...decodeMatches].map(m => parseInt(m[1]));

    // Extract total elapsed_ms (last one)
    const elapsedMatches = log.matchAll(/elapsed_ms=(\d+)/g);
    const elapsedTimes = [...elapsedMatches].map(m => parseInt(m[1]));
    const totalMs = elapsedTimes.length > 0 ? elapsedTimes[elapsedTimes.length - 1] : 0;

    // Extract chunk_samples to calculate total audio duration
    const samplesMatches = log.matchAll(/chunk_samples=(\d+)/g);
    const allSamples = [...samplesMatches].map(m => parseInt(m[1]));
    const totalSamples = allSamples.reduce((a, b) => a + b, 0);

    // Calculate metrics
    const maxGap = frameGaps.length > 0 ? Math.max(...frameGaps) : 0;
    const avgGap = frameGaps.length > 0 ? Math.round(frameGaps.reduce((a, b) => a + b, 0) / frameGaps.length) : 0;
    const p95Gap = frameGaps.length > 0 ? frameGaps.sort((a, b) => a - b)[Math.floor(frameGaps.length * 0.95)] : 0;
    const avgDecode = decodeTimes.length > 0 ? Math.round(decodeTimes.reduce((a, b) => a + b, 0) / decodeTimes.length) : 0;

    // RTF = total_time / audio_duration
    // audio_duration_ms = samples / 24000 * 1000
    const audioDurationMs = (totalSamples / 24000) * 1000;
    const rtf = audioDurationMs > 0 ? (totalMs / audioDurationMs).toFixed(3) : 0;

    // Output
    console.log('METRIC max_frame_gap_ms=' + maxGap);
    console.log('METRIC avg_frame_gap_ms=' + avgGap);
    console.log('METRIC p95_frame_gap_ms=' + p95Gap);
    console.log('METRIC decode_ms_per_frame=' + avgDecode);
    console.log('METRIC total_ms=' + totalMs);
    console.log('METRIC rtf=' + rtf);
    console.log('METRIC frame_count=' + frameGaps.length);
    console.log('METRIC audio_duration_ms=' + Math.round(audioDurationMs));

    console.error('');
    console.error('=== Results ===');
    console.error('Max frame gap: ' + maxGap + 'ms');
    console.error('Avg frame gap: ' + avgGap + 'ms');
    console.error('P95 frame gap: ' + p95Gap + 'ms');
    console.error('Avg decode time: ' + avgDecode + 'ms/frame');
    console.error('Total time: ' + totalMs + 'ms');
    console.error('Audio duration: ' + Math.round(audioDurationMs) + 'ms');
    console.error('RTF: ' + rtf);
    console.error('Frame count: ' + frameGaps.length);
"

# Save full log for debugging
if [[ "${DEBUG:-}" == "1" ]]; then
    echo "Full log saved to: /tmp/lfm2-benchmark-debug.log" >&2
    cp "$LOG_FILE" /tmp/lfm2-benchmark-debug.log
fi

# Note: cleanup trap will remove the temp log file

