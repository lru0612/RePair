#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -m repair run --experiment "$ROOT/configs/experiments/A0_base_14b.toml" "$@"
