#!/usr/bin/env bash
set -euo pipefail

# Format code with ruff
uv run ruff format chameleon/ scripts/
