#!/usr/bin/env bash
# Run the LFM2-Audio python-source-plugin smoke test.
#
# Mirrors examples/python-source-plugin/run.sh. cd into this dir so the
# manifest's `"plugins": ["."]` resolves to this plugin's plugin.toml.
#
# First invocation provisions a uv-managed venv with torch + CUDA +
# liquid-audio + transformers (~5–7 GB). Subsequent runs reuse it.
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

# REMOTEMEDIA_PYTHON_SRC tells env_manager where to find the runner's
# `remotemedia` package to install into the per-plugin venv. Same as
# every other source-load example — see
# examples/python-source-plugin/run.sh for the canonical pattern.
export REMOTEMEDIA_PYTHON_SRC="${REMOTEMEDIA_PYTHON_SRC:-$REPO_ROOT/clients/python}"

# LFM2-Audio's cold start (model download + Mimi codec init) is heavy.
# Bump the per-node call timeout so the first session doesn't trip the
# default 30s guard.
export REMOTEMEDIA_NODE_TIMEOUT_MS="${REMOTEMEDIA_NODE_TIMEOUT_MS:-300000}"

exec env PYTHONPATH="$REPO_ROOT/clients/python" python3 consume.py "$@"
