// Dedicated ncu entry point for the v2 fused-tiled kernel.
//
// The shared bench runs v2 78 times across six shapes plus four accuracy checks,
// so picking a specific launch there means hand-counting --launch-skip, and that
// count breaks whenever kCases or the warmup count changes. This binary launches
// the kernel exactly `iters` times (default 1) on one shape taken from argv, and
// links only v2, so `-k regex:fusedtiling --launch-count 1` is unambiguous.
//
//   profile_v2 <batch> <heads> <seq_len> <head_dim> <causal:0|1> [iters]
//
// Cold cache (algorithmic traffic upper bound, ncu flushes between launches):
//   sudo ncu -k regex:fusedtiling --launch-count 1 \
//       --metrics dram__bytes_read.sum \
//       ./profile_v2 1 8 2048 128 1
//
// Warm cache (steady state, L2 keeps K/V tiles across blocks):
//   sudo ncu -k regex:fusedtiling --launch-skip 4 --launch-count 1 \
//       --cache-control none --metrics dram__bytes_read.sum \
//       ./profile_v2 1 8 2048 128 1 5

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "flash_attention.h"

namespace {

using cuda_op_lab::flash_attention::AttentionShape;
using cuda_op_lab::flash_attention::FusedTilingInfo;
using cuda_op_lab::flash_attention::fused_tiling_hbm_bytes;
using cuda_op_lab::flash_attention::launch_attention_fusedtiling;
using cuda_op_lab::flash_attention::query_fused_tiling;

#define CUDA_CHECK(expr)                                                                                      \
    do {                                                                                                      \
        const cudaError_t status = (expr);                                                                     \
        if (status != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(status), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                                                           \
        }                                                                                                      \
    } while (0)

void usage(const char* argv0)
{
    std::fprintf(stderr,
                 "usage: %s <batch> <heads> <seq_len> <head_dim> <causal:0|1> [iters]\n"
                 "  head_dim must be 64 or 128 (the kernel's template instantiations)\n"
                 "  iters defaults to 1, so ncu --launch-count 1 needs no --launch-skip\n"
                 "  suggested shapes:\n"
                 "    1 8 2048 128 1   head_dim=128, 1 block/SM, latency hiding leans on the pipeline\n"
                 "    2 8 2048  64 1   head_dim=64,  3 blocks/SM, latency hiding leans on occupancy\n",
                 argv0);
}

// Profiling does not care about the values, but all-zero input would push every
// __expf onto an identical path and make the score matrix degenerate. A normal
// distribution keeps the arithmetic representative of real attention inputs.
void fill_normal(std::vector<__nv_bfloat16>& host, unsigned seed)
{
    std::mt19937 engine(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& value : host) value = __float2bfloat16(dist(engine));
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 6 || argc > 7) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    AttentionShape shape{};
    shape.batch = std::atoi(argv[1]);
    shape.heads = std::atoi(argv[2]);
    shape.seq_len = std::atoi(argv[3]);
    shape.head_dim = std::atoi(argv[4]);
    const bool causal = std::atoi(argv[5]) != 0;
    const int iters = (argc == 7) ? std::atoi(argv[6]) : 1;

    if (shape.batch <= 0 || shape.heads <= 0 || shape.seq_len <= 0 || iters <= 0) {
        std::fprintf(stderr, "batch, heads, seq_len and iters must all be positive\n");
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const FusedTilingInfo info = query_fused_tiling(shape.head_dim);
    if (!info.supported) {
        std::fprintf(stderr, "head_dim %d is not instantiated by the v2 kernel\n", shape.head_dim);
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    // Mirror the kernel's own trip count so the report can be matched to a config.
    const int row_blocks = (shape.seq_len + info.rows_per_block - 1) / info.rows_per_block;
    const int last_q = (row_blocks - 1) * info.rows_per_block + info.rows_per_block - 1;
    const int key_limit = causal ? ((last_q < shape.seq_len - 1) ? last_q : shape.seq_len - 1) : (shape.seq_len - 1);
    const int max_loops = (key_limit + info.tile_kv) / info.tile_kv;

    std::printf("device: %s (sm_%d%d)\n", props.name, props.major, props.minor);
    std::printf("shape: batch=%d heads=%d seq=%d head_dim=%d causal=%s\n", shape.batch, shape.heads, shape.seq_len, shape.head_dim, causal ? "yes" : "no");
    std::printf("grid: %d x %d blocks, %d threads/block, %d query rows/block\n", row_blocks, shape.batch * shape.heads, info.block_size, info.rows_per_block);
    std::printf("shared: %d B/block, budget %d B/SM, %d blocks/SM (smem alone would allow %d)\n", info.shared_bytes_per_block, info.device_shared_bytes_per_sm, info.max_blocks_per_sm, info.shared_limited_blocks_per_sm);
    std::printf("registers: %d/thread, local %d B/thread (must be 0, or the accumulator spilled)\n", info.registers_per_thread, info.local_bytes_per_thread);
    std::printf("tile_kv: %d, up to %d K/V tiles per block\n", info.tile_kv, max_loops);
    std::printf("algorithmic HBM: %.3f GB (compare against dram__bytes_read.sum + dram__bytes_write.sum)\n", fused_tiling_hbm_bytes(shape, causal) / 1.0e9);
    std::printf("launching %d time%s\n", iters, iters == 1 ? "" : "s");

    const size_t count = static_cast<size_t>(shape.batch) * shape.heads * shape.seq_len * shape.head_dim;
    const size_t bytes = count * sizeof(__nv_bfloat16);

    std::vector<__nv_bfloat16> host(count);

    __nv_bfloat16* device_q = nullptr;
    __nv_bfloat16* device_k = nullptr;
    __nv_bfloat16* device_v = nullptr;
    __nv_bfloat16* device_o = nullptr;
    CUDA_CHECK(cudaMalloc(&device_q, bytes));
    CUDA_CHECK(cudaMalloc(&device_k, bytes));
    CUDA_CHECK(cudaMalloc(&device_v, bytes));
    CUDA_CHECK(cudaMalloc(&device_o, bytes));

    fill_normal(host, 11u);
    CUDA_CHECK(cudaMemcpy(device_q, host.data(), bytes, cudaMemcpyHostToDevice));
    fill_normal(host, 22u);
    CUDA_CHECK(cudaMemcpy(device_k, host.data(), bytes, cudaMemcpyHostToDevice));
    fill_normal(host, 33u);
    CUDA_CHECK(cudaMemcpy(device_v, host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_o, 0, bytes));

    // No warmup loop: every launch in this process is one the profiler should see.
    for (int iter = 0; iter < iters; ++iter) launch_attention_fusedtiling(device_q, device_k, device_v, device_o, shape, causal, nullptr);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(device_q));
    CUDA_CHECK(cudaFree(device_k));
    CUDA_CHECK(cudaFree(device_v));
    CUDA_CHECK(cudaFree(device_o));

    std::printf("done\n");
    return EXIT_SUCCESS;
}
