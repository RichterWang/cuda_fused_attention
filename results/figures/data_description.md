# Performance data: what exists, how it was measured, how to read it

This directory holds the figures for the final report. Everything here is
generated from committed CSV files; nothing is hand-drawn.

    results/figures/fig1_evolution.{png,svg}        implementation evolution, 4 shapes
    results/figures/fig2_speedup_vs_n.{png,svg}     V3 speedup over V2b / V0 vs sequence length
    results/figures/fig3_grid_fill.{png,svg}        V3 throughput vs grid occupancy, 60 shapes
    results/figures/fig4_v0_stages.{png,svg}        V0 stage breakdown (QK / softmax / PV)
    results/figures/fig5_ablation_summary.{png,svg} controlled ablations, relative to each baseline

    plot/make_figures.py                            regenerates all of them

Every figure is emitted as PNG (200 dpi, for the report) and SVG (vector, for
slides or further editing). matplotlib 3.11.2 lives in the infra_dev conda
environment; the figures are pure plotting, no GPU access, so they regenerate on
any machine that has that environment.

## 1. Measurement environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU, sm_86, 46 SMs |
| Memory | 8 GB GDDR6, 256-bit, 448 GB/s peak |
| Driver / runtime | 580.65.06, CUDA runtime 13.0 (device dump: kernels/ablation/result/device.txt) |
| Compiler | nvcc 13.2.86, -O3 -DNDEBUG, compute_86/sm_86, CMAKE_BUILD_TYPE=Release |
| Clock policy | **not locked**. Laptop part; run-to-run drift of 2-5% is normal |
| Cache policy | warm / uncontrolled; no cache flush between samples |

The 448 GB/s figure matters: the algorithmic roofline in the kernel comments was
originally written against the desktop 3070 Ti (608 GB/s). At 448 GB/s the
8 FLOP/B roof of the 8-row tiled kernels is 3.58 TFLOPS, and V2b measures 3.6.

## 2. Where the numbers come from

Three harnesses, each with its own timing method. Their absolute values are not
interchangeable (see section 7).

**(a) Main grid - bench_flash_attention.** Shapes: B in {1,2,4} x N in
{128,512,1024,2048,4096} x D in {64,128} x {dense, causal} = 60 cases, each run
through V0, V1, V2a, V2b and V3. Per kernel: 5 warmup launches, then 30 launches
each wrapped in its own CUDA-event pair, so the 30 samples are independent rather
than one interval divided by 30. Reported as median, Q1, Q3 and IQR. Readback and
validation happen outside the timed region. Sources:
results/raw/flash_attention/{raw_timings,summary,accuracy}.csv and the frozen
console transcript results/final_performance_benchmark.txt. This run produced 300
summary rows (60 x 5), 9000 raw samples and 20 accuracy rows.

**(b) Four ablation groups - ablation_experiments.** 3-6 representative cases per
group; variants are interleaved round-robin inside the sample loop (30 samples
each), which changes cache and thermal state relative to (a). Sources:
kernels/ablation/result/<group>/{raw,summary,speedup}.csv plus a latency.svg per
group. Correctness in the same program: 8 variants x 280 cases (D in {64,128} x
N in {1,7,8,9,31,32,33,63,64,65,127,128,129,256} x 2 masks x 5 input families),
all PASS.

**(c) V0 stage timing - profile_attention_v0.** Four CUDA events on one stream
(start, after QK, after softmax, after PV), 5 warmups then 30 fresh pipelines,
plus an uninstrumented original-V0 interval per sample as a control. Two runs:

| run | shape | workspace | QK | softmax | PV | pipeline | original V0 |
|---|---|---|---|---|---|---|---|
| 1 | B=2 H=8 N=2048 D=64 dense | 256 MiB | 29.711 | 1.286 | 6.621 | 37.619 | 37.613 |
| 2 | B=1 H=8 N=2048 D=128 dense | 128 MiB | 29.594 | 0.645 | 6.633 | 36.871 | 36.869 |

The two runs happen to have identical FLOPs (2*B*H*N^2*D = 8.59 GFLOP each),
which is what makes the comparison in fig4 meaningful. Source:
results/raw/v0_stage_timing.csv.

## 3. Metric definitions

| column | meaning |
|---|---|
| `median_ms` | median of the 30 per-launch samples |
| `q1_ms`, `q3_ms` | 25th / 75th percentile (type-7 quantile) |
| `iqr_ms` | **IQR = Q3 - Q1**, the width of the middle 50% of samples. Used instead of a standard deviation because one slow launch (thermal blip, clock change) would drag a stddev around. Small IQR means the measurement is reproducible; it says nothing about absolute speed. |
| `tflops` | FLOPs / median time. FLOPs = 4*B*H*N^2*D dense, half that causal |
| `algorithmic_hbm_gb` | **modelled** DRAM traffic, not a profiler reading. Counts whole K/V tiles per query block, so it is an upper bound: measured DRAM is far lower where L2 absorbs the re-reads (95.5% hit rate on the profiled shape) |
| `speedup_vs_v0` | V0 median / kernel median for the same shape and mask |
| row-scaled error | max abs(O - O_ref) / max over the row of abs(O_ref), floored at 1e-6. The pass/fail metric, because attention outputs cancel near zero and per-element relative error explodes there |
| `row_scale_ulp` | abs(O - bf16(O_ref)) / ulp(row max abs(O_ref)), gate 2.0 |

## 4. The five figures

**fig1 - implementation evolution.** Median latency of V0/V1/V2a/V2b/V3 on four
representative shapes, log scale. V3 is 55-83x faster than V0 and 6-9x faster than
V2b. V1 sits below V0 (0.45-0.85x) and that is intended: fusion alone removes the
score matrix but not the per-row K/V re-read, so it is slower.

**fig2 - V3 speedup versus sequence length.** Speedup over V2b and over V0 at
B=1 H=8 dense, D=64 and D=128. The advantage is a function of N, not a constant:
about 2x at N=128 rising to about 9x at N=4096. Reason in fig3.

**fig3 - throughput versus grid occupancy.** All 60 V3 runs; x is launched blocks
divided by resident slots (46 SMs x 3 blocks for D=64, x 2 for D=128). Throughput
saturates at 28-30 TFLOPS once the grid exceeds roughly one wave. At N=128, B=1
only 16 blocks launch (0.12 waves), 30 of 46 SMs idle, and throughput collapses to
3 TFLOPS. This is the honest limitation of the 64-row query tile: right for large
N, wrong for N=128.

**fig4 - V0 stage breakdown.** Left: the three kernels stacked for both runs. The
stages sum to the pipeline in each case (B2 D64: 29.711+1.286+6.621 = 37.618 vs
37.619 measured; B1 D128: 29.594+0.645+6.633 = 36.871 vs 36.871), and the
instrumented pipeline reproduces the uninstrumented original V0 to within 0.02%,
so the event markers do not perturb the run.

Right: the ratio between the two runs, which separates what each stage tracks.
The two runs have equal FLOPs but half as many rows (B=1 vs B=2), and the ratios
come out at **QK 1.00x, PV 1.00x, softmax 0.50x**: QK and PV scale with FLOPs,
while the standalone softmax kernel scales with the number of rows B*H*N. QK
dominates either way (79-80% of the pipeline), which is the expected cost of
materialising the B*H*N*N fp32 score matrix and reading it back twice.

Caveat: the two runs differ in **both** batch and head_dim (the D=128 run used B=1),
so they are not a controlled one-variable comparison. The equal-FLOPs coincidence is
what makes the ratio panel interpretable; a run at B=2 H=8 N=2048 D=128 would make it
a clean design-of-experiments result.

**fig5 - controlled ablations.** Each bar is a variant relative to its own group
baseline, so only within-group comparisons are meaningful. Real wins: b01
(1.29-1.49x), D=128 single-buffered V (1.23x), D=128 asynchronous staging (1.15x).
Neutral: b10 (1.01-1.04x, inside noise), D=64 single buffer (1.00x). Negative:
staging at D=64 is 2-3% slower.

## 5. Headline numbers

| quantity | value |
|---|---|
| V3 vs V0 (60 shapes) | 9.6x min, **57x median**, 86x max |
| V3 vs V2b (60 shapes) | 1.6x min, **7.1x median**, 9.5x max |
| V3 peak throughput | **30.15 TFLOPS** ((2,8,4096,64) dense) |
| V3 median throughput | 24.60 TFLOPS |
| V3 correctness | 20/20 accuracy rows PASS, row-scale ULP 1.00 |
| V3 footprint | D=64 165 regs / 32 KB / 3 blocks per SM; D=128 218 regs / 32 KB / 2 blocks |
| profiled DRAM (2,8,2048,64) | 12.65 MB read + 4.10 MB written = Q+K+V once and O once, the theoretical minimum |
| profiled L2 hit rate | 95.46% |
| tensor pipe occupancy | 42.9% of peak; HMMA count 4,194,304 = 512 blocks x 32 tiles x 4 warps x 64 |
| shared bank conflicts | 18,679 of about 8.4M ldmatrix wavefronts = 0.2% |

## 6. Data quality notes

* No failures anywhere: 0 SKIP, 0 FAIL, 0 NaN/Inf and 0 workspace-allocation
  failures in the main grid; all 8 ablation variants pass the 280-case matrix.
* Timing is tight: median IQR/median across the 60 V3 rows is 1.6%. The worst case
  (13.8%) is N=128, an 11 us kernel sitting on the timer resolution.
* Ablation group A has weak points: its three N=512 rows carry about 20% IQR
  (0.13 ms on a 0.67 ms median). Use them for ratios, not absolute numbers.
* V0 workspace: at N=4096 V0 needs exactly 2.0 GiB. The harness attempts the
  allocation and marks the row skipped if it fails; on this card it succeeded.
* Both V0 stage runs return "correctness: PASS (finite, bitwise equal to original
  V0)", i.e. the instrumented copy is bit-identical, not merely close.

## 7. Do not mix the two harnesses

The main grid and the ablation program measure the same kernels differently
(interleaved variants, different warmup, different sample counts). On
B=4 H=8 N=512 D=64 dense, V2b reads 0.532 ms in the main grid and 0.668 ms in
ablation group A - a 25% gap, with the ablation row also carrying a 20% IQR.
Ratios computed within one harness are reliable; absolute latencies should always
be quoted from the main grid.

## 8. Reproducing

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build build -j

    ./build/bench_flash_attention                       # 60-shape grid -> results/raw/flash_attention/
    ./build/bench_flash_attention_lite                  # 6-shape quick pass, console only
    ./build/ablation_experiments 30 1                   # four ablation groups
    python3 kernels/ablation/summarize.py               # ablation tables + per-group svg
    ./build/profile_attention_v0 2 8 2048 64 0 30 5     # V0 stage breakdown
    # regenerate the five figures (PNG + SVG); matplotlib is in infra_dev
    MPLCONFIGDIR=build/.mplcache /home/minstrel/miniconda3/envs/infra_dev/bin/python plot/make_figures.py

Profiler evidence: results/ncu/v3cold_b2-h8-n2048-d64-c0.ncu-rep (V3),
results/ncu/v2_fullcold_1x8x2048x128.ncu-rep (V2a full set), and 32 reports under
kernels/ablation/result/*/ncu/.

## 9. Known limitations to state in the report

1. Small-N behaviour: V3 needs a filled grid; at N=128 it is only 1.6-2.0x faster
   than V2b because a 64-row tile launches 8x fewer blocks. No tile dispatch by
   sequence length was implemented.
2. Single node, unlocked laptop clocks: no repeated cold-start runs; absolute
   numbers carry a few percent of drift.
3. Algorithmic traffic is a model. Where L2 absorbs the re-reads it overestimates
   DRAM by more than 10x; every claim about actual traffic here is backed by
   Nsight Compute counters, not by that column.
4. V0 stage timing covers two shapes that differ in batch as well as head_dim, so
   the stage ratios are a cross-check rather than a controlled ablation. A B=2
   H=8 N=2048 D=128 run would close that gap in one command.
5. V3b (the occupancy experiment) is not part of the deliverable: it lives in the
   lite benchmark only and is 2-7% slower. The negative result is kept as evidence
   that occupancy is not the remaining lever.
