#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <batch> <heads> <seq_len> <head_dim> <causal:0|1> [iterations]" >&2
  exit 2
fi

build_dir="${BUILD_DIR:-build}"
output_dir="results/ncu"
mkdir -p "$output_dir"

shape="b$1-h$2-n$3-d$4-c$5"
ncu --set full \
  --target-processes all \
  --kernel-name regex:fusedtiling \
  --launch-count 1 \
  --export "$output_dir/v2-$shape" \
  "$build_dir/profile_attention_v2" "$@"
