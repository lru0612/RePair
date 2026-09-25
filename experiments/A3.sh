#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -m repair run --experiment "$ROOT/configs/experiments/A3_induced_14b.toml" "$@"
