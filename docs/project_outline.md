# Memory-Hierarchy-Aware Fused Attention on CUDA GPUs

**Initial project demo and proposal outline**  
**Platform:** NVIDIA CUDA GPU  
**Team size:** Four students  
**Focus:** Parallel reduction, memory hierarchy, shared-memory tiling, and performance attribution

## 1. Project overview

This project studies how a scaled dot-product attention operator can be redesigned for a CUDA GPU by reducing unnecessary data movement and improving data reuse. The operator is developed through three progressively more hardware-aware implementations:

- **V0 — Unfused baseline.** Compute `QK^T`, materialize the score matrix in global memory, apply row-wise stable softmax, and compute `PV` in separate kernels.
- **V1 — Row-fused online-softmax attention.** Assign one query row to a thread block and eliminate the materialized score matrix using online softmax.
- **V2 — Query-tiled fused attention.** Process multiple query rows per block so that a loaded K/V tile is reused across query rows. Candidate mechanisms include vectorized loads/stores, shared-memory swizzling or padding, asynchronous copy, and staged buffering.

The project does not assume that every optimization improves performance. Instead, it uses controlled ablations and profiler evidence to measure the independent and combined effect of each design choice.

### Driving question

> How do query-row reuse, vectorized memory access, shared-memory layout, asynchronous copy, and staged buffering individually and jointly affect fused attention performance on a CUDA GPU?

### Performance objective

The target is to achieve **more than 5x end-to-end speedup over the naive unfused implementation** on selected representative shapes, while preserving the stated numerical-correctness criteria. This is an objective rather than a claimed measured result.

## 2. Scope and computation contract

The operator computes scaled dot-product self-attention:

```math
O = softmax(QK^T / sqrt(D) + M)V.
```

The core scope is:

- Tensor layout: `[batch, heads, sequence_length, head_dimension]`.
- BF16 input/output storage and FP32 accumulation.
- Self-attention with dense and causal masks.
- Head dimensions `D = 64` and `D = 128`.
- A fixed NVIDIA CUDA GPU and a documented CUDA/compiler configuration.
- A scalar CPU FP64 implementation used only as an independent correctness oracle.

The following are out of scope: backward propagation, dropout, training, KV-cache decoding, GQA/MQA, arbitrary masks, complete Transformer inference, multi-GPU execution, and production-library replacement.

## 3. Algorithm evolution

```text
V0: QK^T -> materialized score matrix -> softmax -> PV
  |
  | remove the O(N^2) intermediate
  v
V1: one query row per block + online softmax
  |
  | expose limited K/V reuse and resource constraints
  v
V2: multiple query rows per block + tiled K/V reuse
  |
  | evaluate vectorization, shared-memory layout, async copy, and buffering
  v
Final configuration selected by correctness, latency, and profiler evidence
```

### V0 — Unfused baseline

V0 provides the reference GPU implementation. It uses separate kernels for score computation, stable softmax, and value projection. It materializes an FP32 score workspace of size `B * H * N * N`.

V0 establishes the following measurements:

- End-to-end latency and per-stage latency.
- FLOP count and workspace size.
- Algorithmic global-memory traffic.
- Numerical agreement with the CPU FP64 oracle.

### V1 — Row-fused online-softmax attention

V1 eliminates the score workspace by using an online-softmax state for each query row. It demonstrates that reducing storage alone may not improve end-to-end latency if K/V data is repeatedly loaded for individual query rows or if shared-memory and register use reduce occupancy.

V1 is retained as a deliberate intermediate design, not replaced by V2. It isolates the effect of fusion from the effect of cross-row K/V reuse.

### V2 — Query-tiled fused attention

V2 assigns a tile of query rows to one thread block. A K/V tile loaded from global memory can therefore be reused by multiple query rows. It maintains online-softmax statistics and output accumulators while processing successive K/V tiles.

The V2 design is the basis for a controlled performance study of:

- Multi-row K/V reuse.
- Vectorized BF16 loads and stores.
- Shared-memory padding or swizzling to avoid bank conflicts.
- Asynchronous global-to-shared copy (`cp.async`).
- Single-stage versus multi-stage/shared-memory-buffered execution.

## 4. Correctness requirements

All GPU versions are compared with an independent CPU FP64 oracle. Tests should include:

- Dense and causal attention.
- `D = 64` and `D = 128`.
- Sequence lengths below, equal to, and above tile dimensions.
- Non-multiples of warp and tile sizes, including `N = 1, 31, 32, 33, 63, 64, 65, 127, 128, 129`.
- Zero, one, impulse, random signed, and large-logit inputs.
- NaN/Inf detection.

The report should state an absolute error, a row-scaled error, and a BF16 row-scale ULP rule. Row scaling is important because an attention output can cancel near zero even when the computation producing it is well scaled.

## 5. Controlled V2 ablation study

The complete V2 implementation is treated as a candidate configuration, not an assumed optimum. Each ablation should change one primary factor while keeping shape, input, timing method, and the remaining implementation choices fixed.

| Experiment | Multi-row reuse | Vectorization | Swizzle/padding | Copy path | Buffer staging | Main question |
|---|---:|---:|---:|---|---|---|
| V0 | No | Baseline | No | Synchronous | N/A | What is the cost of materializing scores? |
| V1 | One row | Limited | Optional padding | Synchronous | Single tile | Is fusion alone sufficient? |
| A1 | Disabled or one row | On | On | Same as final V2 | Same as final V2 | What does cross-row K/V reuse contribute? |
| A2 | On | Off | On | Same as final V2 | Same as final V2 | What is the value of vectorized memory access? |
| A3 | On | On | Off | Same as final V2 | Same as final V2 | Does the shared-memory layout avoid meaningful bank conflicts? |
| A4 | On | On | On | Synchronous | Single stage | What is the baseline tiled execution cost? |
| A5 | On | On | On | Asynchronous | Multi-stage if valid | Does asynchronous staging hide loading latency? |

`cp.async` and multi-stage buffering should not be treated as automatically beneficial or as inseparable checkboxes. A synchronous two-buffer variant may add shared-memory pressure without creating overlap. Likewise, if the kernel is compute-bound or occupancy-limited, asynchronous staging may not improve latency. The final configuration should be selected from measured end-to-end results and resource evidence.

## 6. Experimental design

### Core shape grid

| Parameter | Required values |
|---|---|
| Sequence length `N` | 128, 512, 1024, 2048, 4096 |
| Head dimension `D` | 64, 128 |
| Batch/head configuration | `(1, 8)`, `(2, 8)`, `(4, 8)` |
| Mask | Dense and causal |
| Main implementations | V0, V1, and selected V2 |
| Ablations | Two or three representative shapes |

Recommended representative ablation shapes:

- Small-sequence case: `B=4, H=8, N=512, D=64`.
- Long-sequence case: `B=2, H=8, N=2048, D=64`.
- Higher resource-pressure case: `B=1, H=8, N=2048, D=128`.

### Timing and reproducibility

- Preallocate all device buffers.
- Warm up each implementation 5–10 times.
- Collect 20–30 timed repeats and report median plus interquartile range.
- Exclude input generation, file I/O, allocation, and validation from kernel timing.
- Include all kernels and data transformations required by the selected route.
- Store raw data as CSV and generate figures from scripts.
- Record GPU model, compute capability, CUDA version, compiler flags, clock/frequency policy, and build command.

### Performance and hardware evidence

Report:

- End-to-end latency, throughput, TFLOP/s, and speedup.
- V0 workspace size and memory-capacity implications.
- Algorithmic traffic estimates separately from profiler-measured traffic.
- Registers per thread, local-memory spill, shared memory per block, and active blocks/SM.
- Shared-memory bank-conflict evidence where available.
- At least two representative Nsight Compute profiles.

## 7. Course-concept mapping

| Course concept | Project evidence |
|---|---|
| Work and span | Dot-product and softmax reductions; warp/block reduction mapping |
| Memory hierarchy | Global memory, shared memory, and register-resident accumulators |
| Cache-aware blocking | Query/KV tiling and K/V temporal reuse |
| Arithmetic intensity | FLOPs versus data movement across V0/V1/V2 |
| Parallel algorithms | Mapping batch, head, query rows, warps, and lanes to GPU execution |
| Floating-point reduction | Online softmax merge order and numerical error analysis |
| Shared-memory banks | Padding/swizzling ablation and bank-conflict profiling |
| Pipelining | Synchronous versus asynchronous tile staging and potential load/compute overlap |
| Hardware constraints | Register pressure, shared-memory capacity, occupancy, and local-memory spill |

## 8. Team responsibilities

| Member | Primary responsibility | Required evidence | Cross-review |
|---|---|---|---|
| A | CPU FP64 oracle, correctness suite, numerical analysis | Error table and edge-case coverage | V2 causal/tail indexing |
| B | V0/V1, FLOP/workspace/traffic model, fusion analysis | V0/V1 breakdown and fusion result | Benchmark metric definitions |
| C | V2 tiling, multi-row reuse, vectorization, shared-memory layout | Tile/resource sweep and reuse analysis | Oracle/output indexing |
| D | Async-copy/staging experiments, benchmark automation, profiling, plots | Ablation charts and profiler summaries | Synchronization and resource analysis |

Each member should contribute code, raw results, one report figure/table, an interpretation of a hardware result, and one cross-review.

## 9. Six-week work plan

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Freeze API and computation contract; make V0/V1/V2 build and pass small correctness tests | Reproducible baseline and oracle |
| 2 | Complete edge-case tests; validate FLOP, traffic, and resource metrics | Correctness matrix and measurement protocol |
| 3 | Create controlled V2 variants for reuse, vectorization, shared-memory layout, and pipeline studies | Ablation-ready kernels |
| 4 | Run tile sweeps and preliminary Nsight Compute analysis | Frozen candidate configurations |
| 5 | Run full benchmark grid; save raw CSV and generate plots | Results, profiles, and first interpretation |
| 6 | Reproduce from a clean build; complete report and demonstration | Submission package and demo script |

## 10. Demonstration plan

The final demonstration should take approximately ten minutes.

1. **Problem and design evolution (1 minute).** Explain V0, V1, and V2 with one dataflow diagram.
2. **Correctness (2 minutes).** Run a small correctness suite against the CPU FP64 oracle for dense and causal cases.
3. **Representative benchmark (2 minutes).** Run one selected shape and print V0/V1/V2 latency, TFLOP/s, and speedup.
4. **Ablation result (2 minutes).** Present pre-generated plots that isolate multi-row reuse, vectorization, shared-memory layout, and copy/staging effects.
5. **Profiler evidence (2 minutes).** Show pre-collected Nsight Compute summaries for traffic, bank conflicts, occupancy, and resource use. Do not run a lengthy profiler live.
6. **Conclusion (1 minute).** State which mechanisms produced measurable benefit, which did not, and why the final configuration was selected.

## 11. Submission package

```text
README.md
docs/
  project_report.pdf
  demo_slides.pdf
  reproduction.md
results/
  raw/*.csv
  figures/*.png
  profiler/*
scripts/
  run_correctness.sh
  run_quick_demo.sh
  run_full_benchmark.sh
  plot_results.py
operators/transformer_ops/flash_attention/
```

The README should provide a concise project summary, one clean-build command, one correctness command, one quick demonstration command, and the expected output format.
