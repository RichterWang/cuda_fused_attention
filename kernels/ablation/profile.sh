#!/usr/bin/env bash
set -euo pipefail
ncu_bin="${NCU_BIN:-/home/minstrel/miniconda3/envs/infra_dev/bin/ncu}"
out="kernels/ablation/result/b_v_access/ncu"
mkdir -p "$out"
for mode in cold warm; do
  for shape in '2 8 2048 64 0' '1 8 2048 128 1'; do
    read -r b h n d causal <<< "$shape"
    for variant in v2a b01 b10 v2b; do
      if [[ "$mode" == cold ]]; then
        flags=(--cache-control all)
      else
        flags=(--cache-control none --replay-mode application)
      fi
      "$ncu_bin" --set basic "${flags[@]}" --launch-skip 5 --launch-count 1 --force-overwrite --export "$out/${mode}_${variant}_d${d}" ./build/ablation_experiments --profile "$variant" "$b" "$h" "$n" "$d" "$causal"
    done
  done
done
