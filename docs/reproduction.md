# Reproducing the reported results

Every number in the report and the demonstration video comes from the commands
below. Nothing was hand-edited: the CSVs under `results/raw/` and
`kernels/ablation/result/` are the direct output of the binaries, and the figures are
generated from those CSVs.

## 1. Environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop, sm_86, 46 SMs, 8 GB, 448 GB/s |
| Driver / CUDA runtime | 580.65.06 / 13.0 (`kernels/ablation/result/device.txt`) |
| Compiler | nvcc 13.2.86, `-O3 -DNDEBUG`, `compute_86/sm_86` |
| CMake | Release, CUDA separable compilation, C++17 |
| Clocks | not locked. This is a laptop part; 2-5% run-to-run drift is normal |
| Cache | warm / uncontrolled; no cache flush between timed samples |

An RTX 3070 Ti **desktop** has 608 GB/s; the laptop part used here has 448 GB/s.
Roofline statements in the kernel comments use the laptop figure.

## 2. Build

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build build -j

Targets: `flash_attention_v0` ... `flash_attention_v3b` (one static library per
kernel), `bench_flash_attention`, `bench_flash_attention_lite`,
`profile_attention_v0`, `profile_attention_v2`, `ablation_experiments`.
`-DFA_V3_DEBUG_DUMP=ON` additionally makes V3 dump its first tile to
`v3_dump_{q,k,s,p}.bin` for fragment-layout debugging.

## 3. Correctness

    ctest --test-dir build --output-on-failure

Two tests: the full benchmark's own FP64 check and the V0 profiling copy's
bitwise-equivalence check. The full benchmark prints one accuracy row per
kernel per (head_dim, mask) pair at `batch=1 heads=2 seq=256` and fails on
row-scale ULP > 2.0, NaN or Inf.

Boundary coverage beyond that shape lives in two places:

* `bench_flash_attention_lite` sweeps `N in {1, 63, 64, 65, 129, 256}` for
  every kernel, which is where "shorter than one tile", "exact tile" and both
  tail cases are exercised.
* `ablation_experiments` runs 8 variants over `D in {64,128}` x
  `N in {1,7,8,9,31,32,33,63,64,65,127,128,129,256}` x 2 masks x 5 input families
  (280 cases per variant) before any timing. All eight pass.

## 4. Quick demonstration (about 15 seconds of GPU time)

    ./build/bench_flash_attention_lite

Six shapes, five kernels (plus the exploration variant V3b), three warmups and
ten averaged launches. It prints to the terminal only and is **not** the source
of the reported medians. Expect the closing line
`all kernels within their bf16 ULP tolerance at the row scale.`

## 5. Full grid (the reported numbers)

    ./scripts/run_full_benchmark.sh            # -> results/raw/flash_attention/
    # or: ./build/bench_flash_attention /tmp/scratch_grid

60 workloads (3 batch x 5 sequence lengths x 2 head dimensions x 2 masks) x 5
implementations, 5 warmups and 30 individually timed launches each. Expected
output: `summary.csv` 300 rows, `raw_timings.csv` 9000 rows,
`accuracy.csv` 20 rows, and a console transcript with the same content
(`results/final_performance_benchmark.txt` is the archived transcript of the run
cited in the report).

V0 needs a 2.0 GiB fp32 workspace at `N=4096`; the harness attempts the
allocation and marks a row skipped if it fails rather than aborting.

## 6. Controlled ablations

    ./build/ablation_experiments 30 1
    python3 kernels/ablation/summarize.py

Runs correctness first, then 30 interleaved samples per variant, and writes
`kernels/ablation/result/<group>/{raw,summary,speedup}.csv` plus a latency SVG per
group. Four groups: A implementation evolution, B V access, C v2a pipeline,
D v2b key buffering.

## 7. V0 stage timing

    ./build/profile_attention_v0 2 8 2048 64  0 30 5
    ./build/profile_attention_v0 1 8 2048 128 0 30 5
    ./build/profile_attention_v0 --check

Four CUDA events on one stream (start, after QK, after softmax, after PV), 30
fresh pipelines after 5 warmups, plus an uninstrumented original-V0 interval as
a control. The archived table is `results/raw/v0_stage_timing.csv`; the three
stages sum to the pipeline and the instrumented copy matches original V0 to
0.02%.

## 8. Nsight Compute

    bash scripts/profile_ncu.sh 1 8 2048 128 1        # V2a focused entry point
    ncu --set full --cache-control all \
        -k regex:tensorcore_v3 --launch-skip 50 --launch-count 1 \
        --export results/ncu/v3cold_b2-h8-n2048-d64-c0 \
        ./build/bench_flash_attention_lite

The archived captures are `results/ncu/v2_fullcold_1x8x2048x128.ncu-rep` and
`results/ncu/v3cold_b2-h8-n2048-d64-c0.ncu-rep`. Both are large binaries and are
therefore not committed; they stay on the measurement host. Use
`--kernel-name-base demangled` when the filter does not match. `tensorcore_v3` also matches the V3b kernel; use
`regex:tensorcore_v3b` or `regex:tensorcore_v3<` to separate them.

## 9. Figures

    MPLCONFIGDIR=build/.mplcache <python-with-matplotlib> scripts/plot_results.py

Regenerates all five figures in `results/figures/` as PNG (200 dpi) and SVG,
from the CSVs only. matplotlib is not needed to build or run anything else.

## 10. Where the artifacts land

| artifact | produced by | consumed by |
|---|---|---|
| `results/raw/flash_attention/{summary,raw_timings,accuracy}.csv` | §5 | every number in the report |
| `results/final_performance_benchmark.txt` | §5, archived | console transcript cited in the report |
| `results/raw/v0_stage_timing.csv` | §7 | V0 stage breakdown |
| `kernels/ablation/result/` | §6 | ablation tables |
| `results/ncu/*.ncu-rep` | §8 | profiler evidence |
| `results/figures/*.png` + `data_description.md` | §9 | report figures and metric definitions |
| `docs/project_report.pdf` | LaTeX, from a local source package | the submitted report |

The LaTeX source, its figure copies and the evidence manifest live in a local
`tex_workspace/` that is deliberately not redistributed: only the compiled PDF is
submitted. The CSVs the report cites are the ones listed above, under
`results/`; the ablation archive and the Nsight Compute captures stay in
`kernels/ablation/result/` and `results/ncu/` on the measurement host, and are
regenerated by §6 and §8.

## 11. Provenance

* The ablation archive carries its own source metadata
  (`kernels/ablation/result/{commit,nvcc_version,device,source_sha256}.txt`,
  `worktree.patch`) and was produced at commit `5ebfe8a`.
* The main grid and V3 were measured later, in a worktree that added V3 on top of
  that commit. The submission commit recorded in the report is the commit of the
  tree that contains the sources and the archived CSVs; re-verify that the two
  match before submitting.
* The local LaTeX package keeps a manifest (`evidence_manifest.txt`) that
  hashes its copies of the evidence CSVs. It is a convenience for the authors,
  not part of the submission, and the CSVs it duplicates are the ones under
  `results/` listed in section 10.

## 12. How to read the numbers (the two rules that matter)

1. **Do not mix harnesses.** The main grid and `ablation_experiments` time the
   same kernels differently (interleaved variants, different warmup, different
   sample counts). On `B=4 H=8 N=512 D=64` dense, V2b reads 0.532 ms in the main
   grid and 0.668 ms in ablation group A. Ratios within one harness are
   reliable; absolute latencies should be quoted from the main grid.
2. **`algorithmic_hbm_gb` is a model, not a profiler reading.** It counts every
   K/V tile re-read per query block. On `(2,8,2048,64)` it predicts 0.277 GB while
   Nsight Compute measures 12.65 MB read + 4.10 MB written, because 95.5% of the
   re-reads hit in L2. Every claim about actual traffic in the report is backed
   by the profiler counters, not by that column.
