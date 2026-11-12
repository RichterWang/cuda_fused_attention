#!/usr/bin/env bash
# The 60-workload grid (3 batch x 5 sequence lengths x 2 head dims x 2 masks)
# for V0, V1, V2a, V2b and V3. Writes raw, summary and accuracy CSVs.
#
#   ./scripts/run_full_benchmark.sh [build_dir] [output_dir]
#
# Default output_dir is results/raw/flash_attention and existing files there are
# replaced, so pass a scratch directory when you only want to reproduce a run.
set -euo pipefail

build_dir="${1:-build}"
output_dir="${2:-results/raw/flash_attention}"
mkdir -p "$output_dir"
"$build_dir/bench_flash_attention" "$output_dir"
