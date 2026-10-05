#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "flash_attention.h"

namespace {

using cuda_op_lab::flash_attention::AttentionShape;
using cuda_op_lab::flash_attention::attention_flops;
using cuda_op_lab::flash_attention::default_scale;
using cuda_op_lab::flash_attention::FusedRowInfo;
using cuda_op_lab::flash_attention::fused_row_supported;
using cuda_op_lab::flash_attention::launch_attention_fused_row;
using cuda_op_lab::flash_attention::launch_attention_unfused;
using cuda_op_lab::flash_attention::query_fused_row;
using cuda_op_lab::flash_attention::unfused_workspace_bytes;
using cuda_op_lab::flash_attention::FusedTilingInfo;
using cuda_op_lab::flash_attention::launch_attention_fusedtiling;
using cuda_op_lab::flash_attention::query_fused_tiling;
using cuda_op_lab::flash_attention::fused_tiling_hbm_bytes;

constexpr int kWarmupIters = 3;
constexpr int kTimedIters = 10;

// persice check config(example shape)
constexpr int kAccuracyBatch = 1;
constexpr int kAccuracyHeads = 2;
constexpr int kAccuracySeq = 256;

// ULP torlance
constexpr float kUlpTolerance = 2.0f;

// the upper bound of the memory size(on 3070Ti)
constexpr size_t kMaxWorkspaceBytes = 1ull << 30;  // 1 GB

#define CUDA_CHECK(expr)                                                                                       \
    do{                                                                                                        \
        const cudaError_t status = (expr);                                                                     \
        if (status != cudaSuccess)                                                                             \
        {                                                                                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
            std::exit(EXIT_FAILURE);                                                                           \
        }                                                                                                      \
    } while (0)

// case struct 
struct Case
{
    AttentionShape shape;
    bool causal;
    const char* note;
};

// case filling, N represent sequence_length
// N small, show fixed expense; N large, show looping expense
const std::vector<Case> kCases = {
    {{4, 8, 512, 64}, false, "short prompt, dense"},
    {{4, 8, 512, 64}, true, "short prompt, causal"},
    {{2, 8, 2048, 64}, false, "llama-2 context, dense"},
    {{2, 8, 2048, 64}, true, "llama-2 context, causal"},
    {{1, 8, 2048, 128}, true, "llama-7b head_dim, causal"},
    {{1, 8, 4096, 64}, true, "llama-2 max context, causal"},
};

void fill_normal(std::vector<__nv_bfloat16>& host, unsigned seed)
{
    std::mt19937 engine(seed);
    // true distribution of the training value
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& value : host) value = __float2bfloat16(dist(engine));
}

// ulp error caculate, return ulp data
float bf16_ulp(float value)
{
    if (value == 0.0f || !std::isfinite(value)) return std::ldexp(1.0f, -126 - 7); // return 2 * exp（-133）， which is the min ulp

    int exponent = 0;
    std::frexp(std::abs(value), &exponent); // get exponent
    return std::ldexp(1.0f, exponent - 8);
}

// baseline reference kernel
void reference_attention(const std::vector<__nv_bfloat16>& q, const std::vector<__nv_bfloat16>& k, const std::vector<__nv_bfloat16>& v, std::vector<double>& out, const AttentionShape& shape, bool causal)
{
    const int n = shape.seq_len;
    const int d = shape.head_dim;
    const double scale = static_cast<double>(default_scale(d));

    std::vector<double> scores(static_cast<size_t>(n));

    for (int bh = 0; bh < shape.batch * shape.heads; ++bh)
    {
        const size_t slab = static_cast<size_t>(bh) * n * d;

        for (int query = 0; query < n; ++query)
        {
            const int limit = causal ? query + 1 : n;

            double row_max = -1.0e300;
            for (int key = 0; key < limit; ++key)
            {
                double dot = 0.0;
                for (int dim = 0; dim < d; ++dim) dot += static_cast<double>(__bfloat162float(q[slab + static_cast<size_t>(query) * d + dim])) * static_cast<double>(__bfloat162float(k[slab + static_cast<size_t>(key) * d + dim]));
                scores[key] = dot * scale;
                row_max = std::max(row_max, scores[key]);
            }

            double sum = 0.0;
            for (int key = 0; key < limit; ++key)
            {
                scores[key] = std::exp(scores[key] - row_max);
                sum += scores[key];
            }

            for (int dim = 0; dim < d; ++dim)
            {
                double accumulator = 0.0;
                for (int key = 0; key < limit; ++key) accumulator += scores[key] * static_cast<double>(__bfloat162float(v[slab + static_cast<size_t>(key) * d + dim]));
                out[slab + static_cast<size_t>(query) * d + dim] = accumulator / sum;
            }
        }
    }
}

// accurancy data struct
struct Accuracy
{
    float vs_double;
    float ulp;
};

// vs_double as reference, ulp as the comparativate reference
Accuracy compare(const std::vector<__nv_bfloat16>& device_out, const std::vector<double>& reference, const AttentionShape& shape)
{
    Accuracy result{0.0f, 0.0f};

    const int rows = shape.batch * shape.heads * shape.seq_len;
    const int d = shape.head_dim;

    for (int row = 0; row < rows; ++row)
    {
        const size_t base = static_cast<size_t>(row) * d;

        double row_scale = 0.0;
        for (int dim = 0; dim < d; ++dim) row_scale = std::max(row_scale, std::abs(reference[base + dim]));

        const float scale_ulp = bf16_ulp(static_cast<float>(row_scale)); // reference of ulp

        for (int dim = 0; dim < d; ++dim)
        {
            const float got = __bfloat162float(device_out[base + dim]);
            const double want = reference[base + dim];

            const double denominator = std::max(std::abs(want), 1.0e-6);
            result.vs_double = std::max(result.vs_double, static_cast<float>(std::abs(got - want) / denominator));

            const float rounded = __bfloat162float(__float2bfloat16(static_cast<float>(want)));
            result.ulp = std::max(result.ulp, std::abs(got - rounded) / scale_ulp);
        }
    }

    return result;
}

// get DRAM memory traffic
double qkvo_bytes(const AttentionShape& shape)
{
    // Q, K, V read once and O written once, all bf16.
    return 4.0 * shape.batch * shape.heads * shape.seq_len * shape.head_dim * sizeof(__nv_bfloat16);
}

// basline
double unfused_score_bytes(const AttentionShape& shape)
{
    return 5.0 * shape.batch * shape.heads * static_cast<double>(shape.seq_len) * shape.seq_len * sizeof(float);
}

// same handle of each kernel
float time_iterations(void (*body)(void*), void* context)
{
    for (int iter = 0; iter < kWarmupIters; ++iter) body(context);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kTimedIters; ++iter) body(context);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(kTimedIters);
}

// kernel launch param struct
struct Buffers
{
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    __nv_bfloat16* o;
    float* workspace;
    AttentionShape shape;
    bool causal;
};

void run_unfused(void* context)
{
    Buffers& buffers = *static_cast<Buffers*>(context);
    launch_attention_unfused(buffers.q, buffers.k, buffers.v, buffers.o, buffers.workspace, buffers.shape, buffers.causal, nullptr);
}

void run_fused_row(void* context)
{
    Buffers& buffers = *static_cast<Buffers*>(context);
    launch_attention_fused_row(buffers.q, buffers.k, buffers.v, buffers.o, buffers.shape, buffers.causal, nullptr);
}

void run_fused_tiling(void* context)
{
    Buffers& buffers = *static_cast<Buffers*>(context);
    launch_attention_fusedtiling(buffers.q, buffers.k, buffers.v, buffers.o, buffers.shape, buffers.causal, nullptr);    
}

}  // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::printf("device: %s  (sm_%d%d)\n", props.name, props.major, props.minor);
    std::printf("dtype: bf16 storage, fp32 accumulate\n");
    std::printf("HBM GB is the algorithmic traffic, not a profiler reading: v0 includes the fp32 score matrix, fused kernels do not.\n");
    std::printf("accuracy: ulp is the pass/fail number, measured at the row scale (not per element -- attention output cancels), tolerance %.1f\n\n", kUlpTolerance);

    // v1 configuration printing
    std::printf("v1 configuration (local_bytes must be 0, or the output accumulator spilled):\n");
    std::printf("  %10s %10s %8s %14s %8s %8s %12s\n", "head_dim", "block", "tile_n", "shared/block", "regs", "local", "blocks/SM");
    for (int head_dim : {64, 128})
    {
        const FusedRowInfo info = query_fused_row(head_dim);
        if (!info.supported)
        {
            std::printf("  %10d   unsupported\n", head_dim);
            continue;
        }
        std::printf("  %10d %10d %8d %12dB %8d %7dB %12d\n", head_dim, info.block_size, info.tile_n, info.shared_bytes_per_block, info.registers_per_thread, info.local_bytes_per_thread, info.max_blocks_per_sm);
    }
    std::printf("\n");

    // v2 configuration printing
    std::printf("v2 configuration (dynamic shared, bank-conflict-free K reads):\n");
    std::printf("  %10s %10s %8s %8s %14s %8s %8s %12s %12s\n", "head_dim", "block", "tile_kv", "q_rows", "shared/block", "regs", "local", "blocks/SM", "smem-capped");
    for (int head_dim : {64, 128}) {
        const FusedTilingInfo info = query_fused_tiling(head_dim);
        if (!info.supported) {
            std::printf("  %10d   unsupported\n", head_dim);
            continue;
        }
        std::printf("  %10d %10d %8d %8d %12dB %8d %7dB %12d %12d\n",
                    head_dim, info.block_size, info.tile_kv, info.rows_per_block,
                    info.shared_bytes_per_block, info.registers_per_thread,
                    info.local_bytes_per_thread, info.max_blocks_per_sm,
                    info.shared_limited_blocks_per_sm);
    }
    {
        const FusedTilingInfo info = query_fused_tiling(64);
        std::printf("  shared budget: %d B/SM. smem-capped is that budget divided by shared/block;\n", info.device_shared_bytes_per_sm);
        std::printf("  if blocks/SM is lower, registers or the warp limit bind first.\n");
    }
    std::printf("\n");    

    bool all_passed = true;

    // Accuracy, on one small shape, both casual mask settings.
    {
        std::printf("accuracy check (batch=%d heads=%d seq=%d):\n", kAccuracyBatch, kAccuracyHeads, kAccuracySeq);
        std::printf("  %-16s %8s %8s %14s %10s\n", "kernel", "head_dim", "causal", "vs_double", "ulp");

        for (int head_dim : {64, 128})
        {
            for (bool causal : {false, true})
            {
                const AttentionShape shape{kAccuracyBatch, kAccuracyHeads, kAccuracySeq, head_dim};
                const size_t count = static_cast<size_t>(shape.batch) * shape.heads * shape.seq_len * shape.head_dim;
                const size_t bytes = count * sizeof(__nv_bfloat16);

                std::vector<__nv_bfloat16> host_q(count);
                std::vector<__nv_bfloat16> host_k(count);
                std::vector<__nv_bfloat16> host_v(count);
                std::vector<__nv_bfloat16> host_o(count);
                std::vector<double> reference(count);

                fill_normal(host_q, 11u);
                fill_normal(host_k, 22u);
                fill_normal(host_v, 33u);

                reference_attention(host_q, host_k, host_v, reference, shape, causal);

                __nv_bfloat16* device_q = nullptr;
                __nv_bfloat16* device_k = nullptr;
                __nv_bfloat16* device_v = nullptr;
                __nv_bfloat16* device_o = nullptr;
                float* workspace = nullptr;
                CUDA_CHECK(cudaMalloc(&device_q, bytes));
                CUDA_CHECK(cudaMalloc(&device_k, bytes));
                CUDA_CHECK(cudaMalloc(&device_v, bytes));
                CUDA_CHECK(cudaMalloc(&device_o, bytes));
                CUDA_CHECK(cudaMalloc(&workspace, unfused_workspace_bytes(shape)));

                CUDA_CHECK(cudaMemcpy(device_q, host_q.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(device_k, host_k.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(device_v, host_v.data(), bytes, cudaMemcpyHostToDevice));

                Buffers buffers{device_q, device_k, device_v, device_o, workspace, shape, causal};

                struct Entry
                {
                    const char* name;
                    void (*func)(void*);
                    bool (*supported)(int);
                };

                const Entry entries[] = {
                    {"v0_unfused", run_unfused, [](int){return true; }}, 
                    {"v1_fused_row", run_fused_row, fused_row_supported},
                    {"v2_fused_tiled", run_fused_tiling, [](int head_dim){return head_dim == 64 || head_dim == 128 ? true : false;}}
                };

                for (const Entry& entry : entries)
                {
                    if (!entry.supported(head_dim)) continue;

                    CUDA_CHECK(cudaMemset(device_o, 0, bytes));
                    entry.func(&buffers);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(host_o.data(), device_o, bytes, cudaMemcpyDeviceToHost));

                    const Accuracy accuracy = compare(host_o, reference, shape);
                    std::printf("  %-16s %8d %8s %14.6e %10.2f", entry.name, head_dim, causal ? "yes" : "no", accuracy.vs_double, accuracy.ulp);
                    if (accuracy.ulp > kUlpTolerance)
                    {
                        std::printf("   FAIL");
                        all_passed = false;
                    }
                    std::printf("\n");
                }

                CUDA_CHECK(cudaFree(device_q));
                CUDA_CHECK(cudaFree(device_k));
                CUDA_CHECK(cudaFree(device_v));
                CUDA_CHECK(cudaFree(device_o));
                CUDA_CHECK(cudaFree(workspace));
            }
        }
        std::printf("\n");
    }

    // Performance
    for (const Case& item : kCases)
    {
        const AttentionShape& shape = item.shape;
        const size_t count = static_cast<size_t>(shape.batch) * shape.heads * shape.seq_len * shape.head_dim;
        const size_t bytes = count * sizeof(__nv_bfloat16);

        std::vector<__nv_bfloat16> host_input(count);

        __nv_bfloat16* device_q = nullptr;
        __nv_bfloat16* device_k = nullptr;
        __nv_bfloat16* device_v = nullptr;
        __nv_bfloat16* device_o = nullptr;
        CUDA_CHECK(cudaMalloc(&device_q, bytes));
        CUDA_CHECK(cudaMalloc(&device_k, bytes));
        CUDA_CHECK(cudaMalloc(&device_v, bytes));
        CUDA_CHECK(cudaMalloc(&device_o, bytes));

        fill_normal(host_input, 11u);
        CUDA_CHECK(cudaMemcpy(device_q, host_input.data(), bytes, cudaMemcpyHostToDevice));
        fill_normal(host_input, 22u);
        CUDA_CHECK(cudaMemcpy(device_k, host_input.data(), bytes, cudaMemcpyHostToDevice));
        fill_normal(host_input, 33u);
        CUDA_CHECK(cudaMemcpy(device_v, host_input.data(), bytes, cudaMemcpyHostToDevice));

        const size_t workspace_bytes = unfused_workspace_bytes(shape);
        const bool run_v0 = workspace_bytes <= kMaxWorkspaceBytes;

        float* workspace = nullptr;
        if (run_v0) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

        Buffers buffers{device_q, device_k, device_v, device_o, workspace, shape, item.causal};

        const double flops = attention_flops(shape, item.causal);

        std::printf("batch=%d heads=%d seq=%d head_dim=%d causal=%s (%s)\n", shape.batch, shape.heads, shape.seq_len, shape.head_dim, item.causal ? "yes" : "no", item.note);
        std::printf("  %-16s %10s %10s %10s %10s\n", "kernel", "ms", "TFLOPS", "HBM GB", "speedup");

        float baseline_ms = 0.0f;

        if (run_v0)
        {
            baseline_ms = time_iterations(run_unfused, &buffers);
            const double gb = (qkvo_bytes(shape) + unfused_score_bytes(shape)) / 1.0e9;
            std::printf("  %-16s %10.3f %10.2f %10.3f %10s\n", "v0_unfused", baseline_ms, flops / (baseline_ms / 1.0e3) / 1.0e12, gb, "1.00x");
        }
        else
        {
            std::printf("  %-16s %10s %10s %10.3f %10s   (workspace %.1f GB exceeds the %.1f GB cap)\n", "v0_unfused", "-", "-", (qkvo_bytes(shape) + unfused_score_bytes(shape)) / 1.0e9, "-", workspace_bytes / 1.0e9, kMaxWorkspaceBytes / 1.0e9);
        }

        if (fused_row_supported(shape.head_dim))
        {
            const float ms = time_iterations(run_fused_row, &buffers);
            const double kv_reread = 2.0 * shape.batch * shape.heads * static_cast<double>(shape.seq_len) * shape.seq_len * shape.head_dim * sizeof(__nv_bfloat16) * (item.causal ? 0.5 : 1.0);
            const double gb = (qkvo_bytes(shape) + kv_reread) / 1.0e9;

            std::printf("  %-16s %10.3f %10.2f %10.3f", "v1_fused_row", ms, flops / (ms / 1.0e3) / 1.0e12, gb);
            if (run_v0) std::printf(" %9.2fx\n", baseline_ms / ms);
            else std::printf(" %10s\n", "-");
        }

        if (shape.head_dim == 64 || shape.head_dim == 128)
        {
            const float ms = time_iterations(run_fused_tiling, &buffers);
            // v2 shares each K/V tile across the 8 query rows in a block, so the
            // re-read is 1/8 of v1's -- it cannot reuse v1's per-row formula.
            const double gb = fused_tiling_hbm_bytes(shape, item.causal) / 1.0e9;

            std::printf("  %-16s %10.3f %10.2f %10.3f", "v2_fused_tiled", ms, flops / (ms / 1.0e3) / 1.0e12, gb);
            if (run_v0) std::printf(" %9.2fx\n", baseline_ms / ms);
            else std::printf(" %10s\n", "-");
        }

        std::printf("\n");

        CUDA_CHECK(cudaFree(device_q));
        CUDA_CHECK(cudaFree(device_k));
        CUDA_CHECK(cudaFree(device_v));
        CUDA_CHECK(cudaFree(device_o));
        if (run_v0) CUDA_CHECK(cudaFree(workspace));
    }

    if (!all_passed)
    {
        std::printf("FAILED: at least one kernel exceeded %.1f bf16 ULP at the row scale.\n", kUlpTolerance);
        return EXIT_FAILURE;
    }

    std::printf("all kernels within %.1f bf16 ULP at the row scale.\n", kUlpTolerance);
    return EXIT_SUCCESS;
}
