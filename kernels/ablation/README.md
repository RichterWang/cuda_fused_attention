# Attention ablation experiments

- `a_overall`: V0, V2a and V2b end-to-end comparison; the implementations live in `../` and `../v2/`.
- `b_v_access`: V2a, B01, B10 and V2b; B01/B10 are independent V load/layout controls.
- `c_v2a_pipeline`: V2a K synchronous single-buffer, asynchronous single-buffer and asynchronous double-buffer controls.
- `d_v2b_pipeline`: V2b K asynchronous single-buffer versus the existing double-buffer pipeline. V shared-memory layout and packed loads are retained. Buffering and prefetch schedules change together.

`experiments.cu` has its own main and runs correctness before timing. Results are written under `result/<group>/`, with common correctness in `result/correctness.csv`. The entire result directory is Git-ignored.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ablation_experiments -j4
./build/ablation_experiments 30 1
python3 kernels/ablation/summarize.py
bash kernels/ablation/profile.sh
bash kernels/ablation/profile_metrics.sh
bash kernels/ablation/d_v2b_pipeline/profile.sh
```

Run commands from the repository root and serialize GPU measurement/profiling. Set `NCU_BIN` to the local Nsight Compute executable if needed. Timing files are overwritten on rerun; archive them first if previous runs matter. Correctness uses the CPU FP64 oracle for lengths through 256, not the complete benchmark shape grid.
