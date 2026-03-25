#!/bin/bash
set -euo pipefail

# Run tests to ensure correctness
echo "Running tests..." >&2

# Run Rust tests (suppress verbose output, only show failures)
cargo test --release 2>&1 | tail -50

# Check for compilation errors
cargo check --release 2>&1 | grep -E "(error|warning)" || true

echo "Checks passed" >&2
