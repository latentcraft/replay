#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "[1/3] Creating virtual environment with uv..."
uv venv .venv --python 3.10

echo "[2/3] Syncing dependencies from pyproject.toml..."
uv sync

echo "[3/3] Done."
echo "Activate with:"
echo "  source .venv/bin/activate"
