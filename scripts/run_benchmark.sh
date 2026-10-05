#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build}"
"$build_dir/bench_flash_attention"
