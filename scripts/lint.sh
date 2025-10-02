#!/usr/bin/env bash
set -euo pipefail

# Lint code with ruff
uv run ruff check chameleon/ scripts/ "$@"
