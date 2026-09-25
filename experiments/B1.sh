#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -m repair run --experiment "$ROOT/configs/experiments/B1_no_rubric_14b.toml" "$@"
