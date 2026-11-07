#!/usr/bin/env bash
set -euo pipefail
ncu_bin="${NCU_BIN:-/home/minstrel/miniconda3/envs/infra_dev/bin/ncu}"
out=kernels/ablation/result/d_v2b_pipeline/ncu
mkdir -p "$out"
for shape in '2 8 2048 64 0' '2 8 2048 64 1' '1 8 2048 128 0' '1 8 2048 128 1'; do
  read -r b h n d causal <<< "$shape"
  for variant in v2b v2b_single; do
    "$ncu_bin" --set basic --cache-control all --launch-skip 5 --launch-count 1 --force-overwrite --export "$out/${variant}_d${d}_c${causal}" ./build/ablation_experiments --profile "$variant" "$b" "$h" "$n" "$d" "$causal"
    "$ncu_bin" --import "$out/${variant}_d${d}_c${causal}.ncu-rep" --csv --page raw > "$out/${variant}_d${d}_c${causal}.csv"
  done
done
