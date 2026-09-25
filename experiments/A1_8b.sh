#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -m repair run --experiment "$ROOT/configs/experiments/A1_8b_warm_start_8b.toml" "$@"
