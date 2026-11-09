#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <random>
#include <string>
#include <utility>
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
using cuda_op_lab::flash_attention::launch_attention_fusedtiling_v2b;
using cuda_op_lab::flash_attention::query_fused_tiling_v2b;
using cuda_op_lab::flash_attention::fused_tiling_v2b_hbm_bytes;

constexpr int kWarmupIters = 5;
constexpr int kTimedIters = 30;

// persice check config(example shape)
constexpr int kAccuracyBatch = 1;
constexpr int kAccuracyHeads = 2;
constexpr int kAccuracySeq = 256;

// ULP torlance
constexpr float kUlpTolerance = 2.0f;

constexpr double kRowScaleFloor = 1.0e-6;

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

// 30 shapes, each tested with both existing mask modes (60 cases).
const std::vector<Case> kCases = [] {
    std::vector<Case> cases;
    for (int batch : {1, 2, 4})
        for (int seq_len : {128, 512, 1024, 2048, 4096})
            for (int head_dim : {64, 128})
                for (bool causal : {false, true})
                    cases.push_back({{batch, 8, seq_len, head_dim}, causal,
                                     causal ? "core grid, causal" : "core grid, dense"});
    return cases;
}();

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
    return std::ldexp(1.0f, std::max(exponent - 8, -133));
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
    double vs_double = 0.0;
    double row_scaled = 0.0;
    double max_abs = 0.0;
    double ulp = 0.0;
    size_t output_nan = 0;
    size_t output_inf = 0;
    size_t reference_nan = 0;
    size_t reference_inf = 0;

    bool passed() const
    {
        return output_nan == 0 && output_inf == 0 && reference_nan == 0 &&
               reference_inf == 0 && std::isfinite(ulp) && ulp <= kUlpTolerance;
    }
};

// vs_double as reference, ulp as the comparativate reference
Accuracy compare(const std::vector<__nv_bfloat16>& device_out, const std::vector<double>& reference, const AttentionShape& shape)
{
    Accuracy result;

    const int rows = shape.batch * shape.heads * shape.seq_len;
    const int d = shape.head_dim;

    for (int row = 0; row < rows; ++row)
    {
        const size_t base = static_cast<size_t>(row) * d;

        double row_scale = 0.0;
        for (int dim = 0; dim < d; ++dim)
            if (std::isfinite(reference[base + dim]))
                row_scale = std::max(row_scale, std::abs(reference[base + dim]));

        const float scale_ulp = bf16_ulp(static_cast<float>(row_scale)); // reference of ulp

        for (int dim = 0; dim < d; ++dim)
        {
            const float got = __bfloat162float(device_out[base + dim]);
            const double want = reference[base + dim];

            // Check before std::max: a NaN difference can otherwise disappear.
            result.output_nan += std::isnan(got);
            result.output_inf += std::isinf(got);
            result.reference_nan += std::isnan(want);
            result.reference_inf += std::isinf(want);
            if (!std::isfinite(got) || !std::isfinite(want))
            {
                result.vs_double = result.row_scaled = result.max_abs = result.ulp =
                    std::numeric_limits<double>::infinity();
                continue;
            }

            const double abs_error = std::abs(static_cast<double>(got) - want);
            const double denominator = std::max(std::abs(want), 1.0e-6);
            result.vs_double = std::max(result.vs_double, abs_error / denominator);
            result.max_abs = std::max(result.max_abs, abs_error);
            // max_{row,dim} |O - O_ref| / max(max_dim |O_ref|, epsilon).
            // This is independent of BF16 rounding and the ULP metric below.
            result.row_scaled = std::max(result.row_scaled,
                                         abs_error / std::max(row_scale, kRowScaleFloor));

            const float rounded = __bfloat162float(__float2bfloat16(static_cast<float>(want)));
            result.ulp = std::max(result.ulp,
                                 std::abs(static_cast<double>(got) - rounded) / scale_ulp);
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

struct Timing
{
    std::vector<double> raw_ms;  // Preserve measurement order, not sorted order.
    double median_ms = 0.0;
    double q1_ms = 0.0;
    double q3_ms = 0.0;
    double iqr_ms = 0.0;
};

// Linear interpolation at (n - 1) * p (also known as the type-7 quantile).
double quantile(const std::vector<double>& sorted, double p)
{
    const double index = (sorted.size() - 1) * p;
    const size_t lo = static_cast<size_t>(index);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
}

Timing summarize_timings(std::vector<double> raw_ms)
{
    Timing timing;
    timing.raw_ms = std::move(raw_ms);
    auto sorted = timing.raw_ms;
    std::sort(sorted.begin(), sorted.end());
    timing.median_ms = quantile(sorted, 0.5);
    timing.q1_ms = quantile(sorted, 0.25);
    timing.q3_ms = quantile(sorted, 0.75);
    timing.iqr_ms = timing.q3_ms - timing.q1_ms;
    return timing;
}

// Every sample includes the whole route (all three launches for V0).
Timing time_iterations(void (*body)(void*), void* context)
{
    for (int iter = 0; iter < kWarmupIters; ++iter) body(context);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> samples;
    samples.reserve(kTimedIters);
    for (int iter = 0; iter < kTimedIters; ++iter)
    {
        CUDA_CHECK(cudaEventRecord(start));
        body(context);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f)
        {
            std::fprintf(stderr, "Invalid CUDA event timing: %g ms\n", elapsed_ms);
            std::exit(EXIT_FAILURE);
        }
        samples.push_back(elapsed_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return summarize_timings(std::move(samples));
}

std::ofstream open_csv(const std::filesystem::path& path)
{
    std::ofstream file(path);
    if (!file)
    {
        std::fprintf(stderr, "Cannot open CSV: %s\n", path.string().c_str());
        std::exit(EXIT_FAILURE);
    }
    file.imbue(std::locale::classic());
    file << std::setprecision(17);
    return file;
}

void flush_csv(std::ofstream& file)
{
    file.flush();
    if (!file)
    {
        std::fprintf(stderr, "Failed to write benchmark CSV.\n");
        std::exit(EXIT_FAILURE);
    }
}

void csv_key(std::ostream& file, const char* kernel, const AttentionShape& shape, bool causal)
{
    file << kernel << ',' << shape.batch << ',' << shape.heads << ','
         << shape.seq_len << ',' << shape.head_dim << ',' << (causal ? 1 : 0);
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

void run_fused_tiling_v2b(void* context)
{
    Buffers& buffers = *static_cast<Buffers*>(context);
    launch_attention_fusedtiling_v2b(buffers.q, buffers.k, buffers.v, buffers.o, buffers.shape, buffers.causal, nullptr);
}

struct Entry
{
    const char* name;
    void (*func)(void*);
    bool (*supported)(int);
};

const Entry kEntries[] = {
    {"v0_unfused", run_unfused, [](int){ return true; }},
    {"v1_fused_row", run_fused_row, fused_row_supported},
    {"v2a_fused_tiled", run_fused_tiling,
     [](int d){ return query_fused_tiling(d).supported; }},
    {"v2b_fused_tiled", run_fused_tiling_v2b,
     [](int d){ return query_fused_tiling_v2b(d).supported; }}
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc > 2 || (argc == 2 && std::string(argv[1]) == "--help"))
    {
        std::printf("Usage: %s [output_directory]\n"
                    "Default: results/raw/flash_attention (relative to working directory).\n"
                    "Writes raw_timings.csv, summary.csv and accuracy.csv; existing files are replaced.\n",
                    argv[0]);
        return argc > 2 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    const std::filesystem::path output_dir = argc == 2 ? argv[1] : "results/raw/flash_attention";
    std::error_code directory_error;
    std::filesystem::create_directories(output_dir, directory_error);
    if (directory_error)
    {
        std::fprintf(stderr, "Cannot create output directory: %s\n", directory_error.message().c_str());
        return EXIT_FAILURE;
    }
    auto raw_csv = open_csv(output_dir / "raw_timings.csv");
    auto summary_csv = open_csv(output_dir / "summary.csv");
    auto accuracy_csv = open_csv(output_dir / "accuracy.csv");
    raw_csv << "kernel,batch,heads,seq_len,head_dim,causal,iteration,elapsed_ms\n";
    summary_csv << "kernel,batch,heads,seq_len,head_dim,causal,status,warmup_iters,timed_iters,"
                   "median_ms,q1_ms,q3_ms,iqr_ms,baseline_median_ms,speedup_vs_v0,"
                   "tflops,algorithmic_hbm_gb,v0_workspace_bytes,output_nan,output_inf\n";
    accuracy_csv << "kernel,batch,heads,seq_len,head_dim,causal,vs_double,row_scaled_error,"
                    "max_abs_error,row_scale_ulp,output_nan,output_inf,reference_nan,reference_inf,passed\n";

    std::printf("device: %s  (sm_%d%d)\n", props.name, props.major, props.minor);
    std::printf("dtype: bf16 storage, fp32 accumulate\n");
    std::printf("HBM GB is the algorithmic traffic, not a profiler reading: v0 includes the fp32 score matrix, fused kernels do not.\n");
    std::printf("accuracy: finite outputs/reference and row-scale ULP <= %.1f are required.\n", kUlpTolerance);
    std::printf("row_scaled_error = max |O-O_ref| / max(row max |O_ref|, %.1e); max_abs_error = max |O-O_ref|.\n", kRowScaleFloor);
    std::printf("FP64 error metrics use the small accuracy suite; every performance case also checks output NaN/Inf.\n");
    std::printf("timing: %d warmups, %d individual CUDA-event samples; median and IQR (Q3-Q1).\n", kWarmupIters, kTimedIters);
    std::printf("speedup = V0 median / kernel median for the same shape and mask.\n");
    std::printf("CSV output: %s (existing files are replaced)\n\n", output_dir.string().c_str());

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

    std::printf("v2b configuration (independent experiment baseline):\n");
    std::printf("  %10s %10s %8s %8s %14s %8s %8s %12s %12s\n", "head_dim", "block", "tile_kv", "q_rows", "shared/block", "regs", "local", "blocks/SM", "smem-capped");
    for (int head_dim : {64, 128}) {
        const FusedTilingInfo info = query_fused_tiling_v2b(head_dim);
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
    std::printf("\n");

    bool all_passed = true;

    // Accuracy, on one small shape, both casual mask settings.
    {
        std::printf("accuracy check (batch=%d heads=%d seq=%d):\n", kAccuracyBatch, kAccuracyHeads, kAccuracySeq);
        std::printf("  %-16s %8s %8s %12s %12s %12s %8s %7s %7s %7s %7s %6s\n",
                    "kernel", "head_dim", "causal", "vs_double", "row_scaled", "max_abs",
                    "ulp", "out_nan", "out_inf", "ref_nan", "ref_inf", "status");

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

                for (const Entry& entry : kEntries)
                {
                    if (!entry.supported(head_dim)) continue;

                    // BF16 0xffff is NaN: unwritten output elements must fail.
                    CUDA_CHECK(cudaMemset(device_o, 0xff, bytes));
                    entry.func(&buffers);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(host_o.data(), device_o, bytes, cudaMemcpyDeviceToHost));

                    const Accuracy accuracy = compare(host_o, reference, shape);
                    const bool passed = accuracy.passed();
                    all_passed = all_passed && passed;
                    std::printf("  %-16s %8d %8s %12.4e %12.4e %12.4e %8.2f %7zu %7zu %7zu %7zu %6s\n",
                                entry.name, head_dim, causal ? "yes" : "no", accuracy.vs_double,
                                accuracy.row_scaled, accuracy.max_abs, accuracy.ulp,
                                accuracy.output_nan, accuracy.output_inf, accuracy.reference_nan,
                                accuracy.reference_inf, passed ? "PASS" : "FAIL");
                    csv_key(accuracy_csv, entry.name, shape, causal);
                    accuracy_csv << ',' << accuracy.vs_double << ',' << accuracy.row_scaled << ','
                                 << accuracy.max_abs << ',' << accuracy.ulp << ',' << accuracy.output_nan << ','
                                 << accuracy.output_inf << ',' << accuracy.reference_nan << ','
                                 << accuracy.reference_inf << ',' << (passed ? 1 : 0) << '\n';
                    flush_csv(accuracy_csv);
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
    size_t skipped_runs = 0;
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
        // The largest required V0 workspace is 2 GiB. Try the allocation
        // instead of omitting required shapes because of the old 1 GiB cap.
        float* workspace = nullptr;
        const cudaError_t workspace_status = cudaMalloc(&workspace, workspace_bytes);
        const bool run_v0 = workspace_status == cudaSuccess;
        if (workspace_status == cudaErrorMemoryAllocation)
        {
            // Clear this recoverable error before timing the fused kernels.
            const cudaError_t last_error = cudaGetLastError();
            if (last_error != cudaErrorMemoryAllocation) CUDA_CHECK(last_error);
            std::fprintf(stderr, "V0 workspace allocation failed: B=%d H=%d N=%d D=%d, %.1f MiB.\n",
                         shape.batch, shape.heads, shape.seq_len, shape.head_dim,
                         workspace_bytes / (1024.0 * 1024.0));
        }
        else CUDA_CHECK(workspace_status);

        Buffers buffers{device_q, device_k, device_v, device_o, workspace, shape, item.causal};

        const double flops = attention_flops(shape, item.causal);

        std::printf("batch=%d heads=%d seq=%d head_dim=%d causal=%s (%s)\n", shape.batch, shape.heads, shape.seq_len, shape.head_dim, item.causal ? "yes" : "no", item.note);
        std::printf("  %-16s %10s %10s %10s %10s %10s %7s %7s\n",
                    "kernel", "median ms", "IQR ms", "TFLOPS", "HBM GB", "speedup", "NaN", "Inf");

        double baseline_ms = 0.0;
        std::vector<__nv_bfloat16> host_output(count);
        const double kv_reread = 2.0 * shape.batch * shape.heads * static_cast<double>(shape.seq_len) *
                                shape.seq_len * shape.head_dim * sizeof(__nv_bfloat16) * (item.causal ? 0.5 : 1.0);
        const double traffic_gb[] = {
            (qkvo_bytes(shape) + unfused_score_bytes(shape)) / 1.0e9,
            (qkvo_bytes(shape) + kv_reread) / 1.0e9,
            fused_tiling_hbm_bytes(shape, item.causal) / 1.0e9,
            fused_tiling_v2b_hbm_bytes(shape, item.causal) / 1.0e9
        };

        for (size_t index = 0; index < sizeof(kEntries) / sizeof(kEntries[0]); ++index)
        {
            const Entry& entry = kEntries[index];
            const char* skip_reason = !entry.supported(shape.head_dim) ? "unsupported" :
                                      (index == 0 && !run_v0 ? "workspace_allocation_failed" : nullptr);
            if (skip_reason)
            {
                ++skipped_runs;
                std::printf("  %-16s SKIP (%s)\n", entry.name, skip_reason);
                csv_key(summary_csv, entry.name, shape, item.causal);
                summary_csv << ',' << skip_reason << ",0,0";
                // Seven unavailable columns: median through TFLOPS.
                for (int column = 0; column < 7; ++column) summary_csv << ',';
                summary_csv << ',' << traffic_gb[index] << ',' << workspace_bytes << ",,\n";
                flush_csv(summary_csv);
                continue;
            }

            CUDA_CHECK(cudaMemset(device_o, 0xff, bytes));
            const Timing timing = time_iterations(entry.func, &buffers);
            // Readback and validation are deliberately outside CUDA-event timing.
            CUDA_CHECK(cudaMemcpy(host_output.data(), device_o, bytes, cudaMemcpyDeviceToHost));
            size_t output_nan = 0;
            size_t output_inf = 0;
            for (const auto value : host_output)
            {
                const float output = __bfloat162float(value);
                output_nan += std::isnan(output);
                output_inf += std::isinf(output);
            }
            const bool finite = output_nan == 0 && output_inf == 0;
            all_passed = all_passed && finite;
            if (index == 0 && finite) baseline_ms = timing.median_ms;
            const bool have_speedup = baseline_ms > 0.0 && finite;
            const double tflops = flops / (timing.median_ms / 1.0e3) / 1.0e12;

            std::printf("  %-16s %10.4f %10.4f %10.2f %10.3f",
                        entry.name, timing.median_ms, timing.iqr_ms, tflops, traffic_gb[index]);
            if (have_speedup) std::printf(" %9.2fx", baseline_ms / timing.median_ms);
            else std::printf(" %10s", "-");
            std::printf(" %7zu %7zu%s\n", output_nan, output_inf, finite ? "" : " FAIL");

            for (size_t iter = 0; iter < timing.raw_ms.size(); ++iter)
            {
                csv_key(raw_csv, entry.name, shape, item.causal);
                raw_csv << ',' << iter + 1 << ',' << timing.raw_ms[iter] << '\n';
            }
            csv_key(summary_csv, entry.name, shape, item.causal);
            summary_csv << ',' << (finite ? "ok" : "nonfinite_output") << ',' << kWarmupIters << ','
                        << timing.raw_ms.size() << ',' << timing.median_ms << ',' << timing.q1_ms << ','
                        << timing.q3_ms << ',' << timing.iqr_ms << ',';
            if (baseline_ms > 0.0) summary_csv << baseline_ms;
            summary_csv << ',';
            if (have_speedup) summary_csv << baseline_ms / timing.median_ms;
            summary_csv << ',' << tflops << ',' << traffic_gb[index] << ',' << workspace_bytes << ','
                        << output_nan << ',' << output_inf << '\n';
            flush_csv(raw_csv);
            flush_csv(summary_csv);
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
        std::printf("FAILED: row-scale ULP exceeded %.1f or NaN/Inf was detected. CSV files retain the failures.\n", kUlpTolerance);
        return EXIT_FAILURE;
    }

    std::printf("All executed small-case accuracy checks passed (row-scale ULP <= %.1f); all executed performance outputs are finite.\n", kUlpTolerance);
    std::printf("Saved CSV files to %s; %zu kernel/shape runs skipped (see summary.csv status).\n",
                output_dir.string().c_str(), skipped_runs);
    return EXIT_SUCCESS;
}
