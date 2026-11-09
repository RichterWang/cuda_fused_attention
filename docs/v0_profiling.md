# V0 stage profiling

`profile_attention_v0` is the shared entry point for stage timing, checking the
profiling copy against V0, and focused Nsight Compute collection. It does not
change `kernels/v0_unfused.cu` or the existing benchmark.

## Build and run

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target profile_attention_v0 -j

# Check both timing-enabled and event-free copies against original V0:
ctest --test-dir build -R '^v0_profile_correctness$' --output-on-failure
# Equivalent: ./build/profile_attention_v0 --check

# Default: two dense shapes, (2,8,2048,64) and (1,8,2048,128).
./build/profile_attention_v0

# B H N D causal [samples=30 [warmup=5]]
./build/profile_attention_v0 2 8 2048 64 0 30 5
./build/profile_attention_v0 1 8 2048 128 0 30 5
# Causal is also supported:
./build/profile_attention_v0 1 8 2048 128 1 30 5
```

The correctness mode covers N=1,17,256, D=64,128, two heads, and both masks.
It requires finite, bitwise-equal output relative to the original V0 on a
non-default stream. This establishes equivalence of the copy; CPU-oracle testing
remains in the existing benchmark. CUDA errors or mismatches return failure.

## Timing interpretation

The independent `v0_unfused_profile.cu` copies all three V0 kernel bodies and
launch configurations. It records four timing-enabled CUDA Events on the same
stream, at the start, after QK, after softmax, and after PV. There is no
synchronization between stages. Each sample runs a fresh complete pipeline;
softmax therefore always consumes fresh scores, rather than an already
normalized workspace.

Output reports median, Q1, Q3 and IQR in milliseconds for:

- `QK`: score computation, scale and optional causal masking.
- `softmax`: in-place row normalization.
- `PV`: probability/value product and BF16 output.
- `instrumented_pipeline`: the interval across all three stages of the copy.
- `original_v0_e2e`: an independent measurement of the original three-kernel
  pipeline, with events only outside the pipeline.

For each sample, the original pipeline is measured first, then the profiling
copy. Allocation, input initialization, correctness checks and warmups are
outside measurement. These are warm/uncontrolled-cache measurements: no cache
flush or clock locking is performed. The fixed original-then-copy order can
favor the copy's cache state. Stage intervals include event/launch overhead and
must be used as a breakdown, not a replacement for original-V0 end-to-end
latency. Medians of individual stages need not sum to the pipeline median.
For small shapes, event and host launch overhead can dominate.

## Focused NCU collection

`--ncu B H N D causal [iterations=1]` disables validation, warmups and timing
events. Every iteration launches exactly QK, softmax, PV, in order. Inputs are
initialized before launch; CUDA allocation and final synchronization remain
outside the kernels. The copied kernels have distinct, stable names:

```text
v0_profile_qk_kernel
v0_profile_softmax_kernel
v0_profile_pv_kernel
```

Capture all three phases in one report (create the destination first):

```bash
mkdir -p results/ncu
ncu --set full --cache-control all \
  -k 'regex:v0_profile_.*_kernel' --launch-count 3 \
  -o results/ncu/v0_cold_2x8x2048x64_dense \
  ./build/profile_attention_v0 --ncu 2 8 2048 64 0
```

For a selected warm-cache stage, choose its name and skip four matching
launches (not four complete pipelines). Dependencies are still executed:

```bash
ncu --cache-control none -k regex:v0_profile_softmax_kernel \
  --launch-skip 4 --launch-count 1 --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  -o results/ncu/v0_softmax_warm_1x8x2048x128_dense \
  ./build/profile_attention_v0 --ncu 1 8 2048 128 0 5
```

Profiler replay/cache policy affects the workload. Use the timing mode for
latency statistics and NCU for hardware metrics; record shape, mask, device,
CUDA/NCU versions and cache policy with experimental results.
