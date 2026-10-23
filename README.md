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
| `v2b_tiled_pipelined` | Independent copy of the current tiled kernel, built and run separately | Baseline for the next V2 experiment iteration |

Inputs and outputs are BF16; intermediate accumulation is FP32. The supported
attention layouts are `[batch, heads, sequence_length, head_dimension]`, with
dense and causal masks.

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
It runs both V2a and V2b through the same correctness and performance cases.

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
kernels/       V0, V1, V2, and future controlled ablation kernels
benchmarks/    Shared benchmark and focused profiler entry points
scripts/       Reproducible build, benchmark, and NCU commands
configs/       Shape lists and profiler metric sets
results/       Generated raw data, figures, and NCU reports (ignored by Git)
docs/          Project scope, experiment protocol, and demo material
```

## Planned ablations

1. Single query row versus multiple query rows per block.
2. Scalar versus vectorized loads/stores.
3. Shared-memory swizzle/padding off versus on.
4. Synchronous single-stage execution versus asynchronous staged execution.

`cp.async` and multi-stage buffering are evaluated as candidate mechanisms,
not assumed to improve end-to-end performance. See
[the project outline](docs/project_outline.md) for the full experiment design.
