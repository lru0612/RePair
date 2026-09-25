#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -m repair run --experiment "$ROOT/configs/experiments/C2_top5_14b.toml" "$@"
