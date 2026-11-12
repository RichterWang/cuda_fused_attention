#!/usr/bin/env bash
# Historical name; the documented entry point is run_full_benchmark.sh.
set -euo pipefail
exec "$(dirname "$0")/run_full_benchmark.sh" "$@"
