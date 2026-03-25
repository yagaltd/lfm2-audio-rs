#!/bin/bash
set -euo pipefail

# Configuration
MODEL_DIR="${LFM2_MODEL_DIR:-/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX}"
SERVER_BIN="${SERVER_BIN:-./target/release/lfm2-server}"
PORT="${PORT:-8080}"
TIMEOUT="${TIMEOUT:-60}"
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
RUST_LOG=info,lfm2_audio=info "$SERVER_BIN" --model "$MODEL_DIR" --bind "127.0.0.1:$PORT" > "$LOG_FILE" 2>&1 &
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

    echo "Running query: $query_label" >&2

    # Send WebSocket message and wait for completion
    # The server logs will contain the timing info
    node -e "
        const WebSocket = require('ws');
        const ws = new WebSocket('ws://localhost:$PORT/ws');

        ws.on('open', () => {
            ws.send(JSON.stringify({type: 'user.text', text: '$query'}));
        });

        let receivedComplete = false;
        ws.on('message', (data) => {
            const msg = JSON.parse(data.toString());
            if (msg.type === 'assistant.turn') {
                receivedComplete = true;
                ws.close();
            }
        });

        // Timeout after ${TIMEOUT}s
        setTimeout(() => {
            if (!receivedComplete) {
                console.error('Query timed out');
                ws.close();
                process.exit(1);
            }
        }, ${TIMEOUT}000);
    " 2>&1 || {
        echo "ERROR: Query '$query_label' failed" >&2
        return 1
    }

    echo "Query '$query_label' completed" >&2
}

# Run test queries
echo "" >&2
echo "=== Running benchmark queries ===" >&2
run_query "Hello" "short_hello"
sleep 1  # Brief pause between queries
run_query "Tell me a joke" "long_joke"

# Give server a moment to flush logs
sleep 1

# Stop server to ensure all logs are flushed
echo "Stopping server..." >&2
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

# Parse logs and extract metrics
echo "" >&2
echo "=== Parsing timing metrics ===" >&2

# Extract frame_gap_ms values from logs
# Format: "audio frame generated" logs contain frame_gap_ms
# Format: "streaming audio chunk ready" logs contain decode_elapsed_ms

# Filter to relevant log lines
grep -E "(audio frame generated|streaming audio chunk ready|assistant\.turn)" "$LOG_FILE" > "${LOG_FILE}.filtered" 2>/dev/null || true

# Extract metrics using awk
read -r max_frame_gap avg_frame_gap decode_ms total_ms rtf < <(
    node -e "
        const fs = require('fs');
        const lines = fs.readFileSync('${LOG_FILE}.filtered', 'utf8').split('\n');

        const frameGaps = [];
        const decodeTimes = [];
        let streamStartTime = null;
        let lastChunkTime = null;
        let totalAudioSamples = 0;

        for (const line of lines) {
            // Parse frame_gap_ms from 'audio frame generated' logs
            const gapMatch = line.match(/frame_gap_ms=(\d+)/);
            if (gapMatch && parseInt(gapMatch[1]) > 0) {
                frameGaps.push(parseInt(gapMatch[1]));
            }

            // Parse decode_elapsed_ms from 'streaming audio chunk ready' logs
            const decodeMatch = line.match(/decode_elapsed_ms=(\d+)/);
            if (decodeMatch) {
                decodeTimes.push(parseInt(decodeMatch[1]));
            }

            // Parse elapsed_ms for first chunk timing
            const elapsedMatch = line.match(/elapsed_ms=(\d+)/);
            if (elapsedMatch && !streamStartTime) {
                streamStartTime = parseInt(elapsedMatch[1]);
            }
            if (elapsedMatch) {
                lastChunkTime = parseInt(elapsedMatch[1]);
            }

            // Parse chunk_samples for RTF calculation
            const samplesMatch = line.match(/chunk_samples=(\d+)/);
            if (samplesMatch) {
                totalAudioSamples += parseInt(samplesMatch[1]);
            }
        }

        // Calculate metrics
        const maxGap = frameGaps.length > 0 ? Math.max(...frameGaps) : 0;
        const avgGap = frameGaps.length > 0 ? Math.round(frameGaps.reduce((a, b) => a + b, 0) / frameGaps.length) : 0;
        const avgDecode = decodeTimes.length > 0 ? Math.round(decodeTimes.reduce((a, b) => a + b, 0) / decodeTimes.length) : 0;
        const totalTime = lastChunkTime || 0;

        // RTF = total_time / audio_duration
        // audio_duration = samples / sample_rate (24kHz)
        const audioDurationMs = (totalAudioSamples / 24000) * 1000;
        const rtf = audioDurationMs > 0 ? (totalTime / audioDurationMs).toFixed(2) : 0;

        // Output as space-separated values
        console.log(maxGap, avgGap, avgDecode, totalTime, rtf);
    "
)

# Output structured metrics for autoresearch
echo ""
echo "=== Results ===" >&2
echo "METRIC max_frame_gap_ms=$max_frame_gap"
echo "METRIC avg_frame_gap_ms=$avg_frame_gap"
echo "METRIC decode_ms_per_frame=$decode_ms"
echo "METRIC total_ms=$total_ms"
echo "METRIC rtf=$rtf"
echo ""
echo "Max frame gap: ${max_frame_gap}ms" >&2
echo "Avg frame gap: ${avg_frame_gap}ms" >&2
echo "Decode time: ${decode_ms}ms/frame" >&2
echo "Total time: ${total_ms}ms" >&2
echo "RTF: $rtf" >&2

# Save full log for debugging
if [[ "${DEBUG:-}" == "1" ]]; then
    echo "Full log saved to: $LOG_FILE" >&2
    cp "$LOG_FILE" /tmp/lfm2-benchmark-debug.log
else
    rm -f "$LOG_FILE"
fi
