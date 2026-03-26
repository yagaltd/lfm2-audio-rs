#!/bin/bash
set -euo pipefail

cargo test --release --features server --bin lfm2-server 2>&1 | tail -80
node static/assistant-stream-guard.test.mjs
