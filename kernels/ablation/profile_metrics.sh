#!/usr/bin/env bash
set -euo pipefail
ncu_bin="${NCU_BIN:-/home/minstrel/miniconda3/envs/infra_dev/bin/ncu}"
metrics='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,dram__bytes_read.sum,dram__bytes_write.sum'
out=kernels/ablation/result/b_v_access/ncu
mkdir -p "$out"
for shape in '2 8 2048 64 0' '1 8 2048 128 1'; do
  read -r b h n d causal <<< "$shape"
  for variant in v2a b01 b10 v2b; do
    "$ncu_bin" --metrics "$metrics" --cache-control all --launch-skip 5 --launch-count 1 --force-overwrite --export "$out/metrics_${variant}_d${d}" ./build/ablation_experiments --profile "$variant" "$b" "$h" "$n" "$d" "$causal"
    "$ncu_bin" --import "$out/metrics_${variant}_d${d}.ncu-rep" --csv --page raw > "$out/metrics_${variant}_d${d}.csv"
  done
done
