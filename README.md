# Memory-Hierarchy-Aware Fused Attention on CUDA GPUs

This repository studies scaled dot-product attention as a CUDA kernel-design
problem. It progresses from an unfused baseline to an online-softmax fused
kernel and a query-tiled implementation that reuses K/V tiles across multiple
query rows.

The project focuses on controlled performance attribution rather than treating
all optimizations as automatically beneficial. Planned ablations isolate
multi-row reuse, vectorized memory access, shared-memory layout, and
asynchronous staging.

## Implementations

| Variant | Description | Purpose |
|---|---|---|
| `v0_unfused` | Separate `QK^T`, stable softmax, and `PV` kernels; materializes scores | GPU correctness and performance baseline |
| `v1_fused_row` | One query row per block with online softmax | Isolates fusion without cross-row K/V reuse |
| `v2a_tiled_pipelined` | Multiple query rows per block, shared-memory K/V tiles, and pipelined staging | Existing V2 implementation |
| `v2b_tiled_pipelined` | Independent copy of the current tiled kernel, built and run separately | Packed V reads and the single-key-buffer schedule |
| `v3_tensorcore` | BF16 Tensor Core `mma.m16n8k16`; 64 query rows per block, online softmax reduced inside a quad | Arithmetic-level change on top of the tiled design |

Inputs and outputs are BF16; intermediate accumulation is FP32. The supported
attention layouts are `[batch, heads, sequence_length, head_dimension]`, with
dense and causal masks.

`v3b_tensorcore` is a schedule variant of V3 (one `__launch_bounds__` hint that
lets three blocks fit per SM at head dimension 128). It is built and registered
in the lite benchmark only: it measured 2-7% slower than V3 on every shape, so
the main grid does not include it. The negative result is kept on purpose.

## Build

The default target is built for `sm_86`. Override it when building for another
GPU architecture.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

## Correctness and benchmark

```bash
ctest --test-dir build --output-on-failure
./build/bench_flash_attention
```

The benchmark contains a CPU FP64 reference for correctness and reports
end-to-end latency, TFLOP/s, algorithmic traffic, and resource information.
It runs V0, V1, V2a, V2b and V3 through the same correctness and performance
cases: 3 batch settings x 5 sequence lengths x 2 head dimensions x 2 masks =
60 workloads per implementation, 5 warmups and 30 individually timed launches
each, reported as median and IQR. Results land in `results/raw/flash_attention/`.

A shorter entry point runs six representative shapes with three warmups and ten
averaged launches. It is a smoke test, not the source of the reported numbers:

```bash
./build/bench_flash_attention_lite
```

## Results

Measured on an RTX 3070 Ti Laptop (sm_86, 46 SMs, 448 GB/s), nvcc 13.2,
`-O3 -DNDEBUG`, unlocked laptop clocks:

| quantity | value |
|---|---|
| V3 versus V0 | 9.6x min, **57x median**, 86x max over the 60 workloads |
| V3 versus V2b | 1.6x min, **7.1x median**, 9.5x max |
| V3 throughput | 30.2 TFLOPS peak, 24.6 TFLOPS median |
| V3 correctness | 20/20 FP64 accuracy rows pass, row-scale ULP 1.00 |
| measured DRAM (2,8,2048,64) | 12.65 MB read + 4.10 MB written = Q, K, V once and O once |

Figures, the metric definitions and the caveats that belong with these numbers
are in [results/figures](results/figures), with the data itself in
[results/raw/flash_attention](results/raw/flash_attention). Start with
[data_description.md](results/figures/data_description.md): it also explains why
the algorithmic traffic column is an upper bound rather than a profiler reading.

## V0 stage timing

An independent V0 profiling copy measures QK, softmax and PV without changing
original V0. One entry point provides timing, correctness checks and NCU mode:

```bash
cmake --build build --target profile_attention_v0 -j
./build/profile_attention_v0 --check
./build/profile_attention_v0 2 8 2048 64 0 30 5
```

Arguments are `B H N D causal [samples [warmup]]`; no arguments runs two
representative dense shapes. Output includes stage median/IQR and a separate
original-V0 end-to-end measurement. See [V0 profiling](docs/v0_profiling.md)
for measurement details and NCU commands.

## Focused Nsight Compute profiling

The focused profile binary launches only V2. This avoids depending on the
benchmark's launch ordering when using Nsight Compute.

```bash
bash scripts/profile_ncu.sh 2 8 2048 64 1
```

The final argument list is:

```text
<batch> <heads> <seq_len> <head_dim> <causal:0|1> [iterations]
```

## Repository layout

```text
include/       Public shapes, launch interfaces, and metric helpers
kernels/       V0, V1, V2a, V2b, V3, V3b and the controlled ablation kernels
benchmarks/    Shared benchmark, lite benchmark, and focused profiler entries
scripts/       Reproducible build, benchmark, profiling and plotting commands
plot/          Figure generation from the CSVs (matplotlib)
configs/       Shape lists and profiler metric sets
results/       Raw data, figures and profiler reports (see results/figures)
docs/          Scope, experiment protocol, profiling notes, reproduction guide
               and the compiled report (project_report.pdf)
```

## Controlled ablations

Four groups, each isolating one change against its own baseline; the variants
live under [kernels/ablation](kernels/ablation) and the results under
`kernels/ablation/result/`.

| Group | Question | Measured |
|---|---|---|
| A | Implementation evolution | V2a 5.3x, V2b 8.1x over V0 |
| B | Which V-access change matters | packed reads 1.29-1.49x; swizzle alone 1.01-1.04x (inside noise) |
| C | Synchronous versus asynchronous staging | 1.15x at D=128; 0.97x (slower) at D=64 |
| D | Single versus double key buffer | 1.23x at D=128 (55.3 -> 38.9 KB shared, 1 -> 2 blocks); flat at D=64 |

The point of the C and D results is that neither `cp.async` nor extra buffers is
a win by itself: staging only pays where shared memory, not arithmetic, limits
residency. See [the project outline](docs/project_outline.md) for the protocol
and [data_description.md](results/figures/data_description.md) for how the
numbers were produced.
