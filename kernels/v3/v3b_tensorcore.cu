// =============================================================================
// V3b: V3 with exactly one change -- a launch-bounds hint that lets d=128 run
// three blocks per SM instead of two.
//
//   V3  d=64 : 165 registers, 32 KB shared -> 3 blocks (12 warps)
//   V3  d=128: 218 registers, 32 KB shared -> 2 blocks ( 8 warps)
//   V3b d=64 : 150 registers, 32 KB shared -> 3 blocks (12 warps)   [same occupancy]
//   V3b d=128: 161 registers, 32 KB shared -> 3 blocks (12 warps)   [+50% warps]
//
// d=64 keeps the same occupancy as V3, so the six benchmark shapes split into a
// control group (four d=64 shapes, expected to match V3 within run-to-run noise)
// and a treatment group (one d=128 shape, the only one whose residency changes).
// It is not a bit-exact control: merely having the attribute tightens ptxas'
// allocation from 165 to 150 registers at d=64, even though the 3-block floor is
// already met there, and no C++ construct can apply an attribute to only one
// template instantiation. Same occupancy, slightly different schedule.
//
// Everything below the launch_bounds attribute is v3_tensorcore.cu verbatim: same
// XOR swizzle, same K/V double buffering, same split waits (wait<2> for K,
// wait<1> for V), same fragment math, same always-on masking, same numerics. The
// outputs must be bit-identical to V3.
//
// ---------------------------------------------------------------------------
// What was tried first, and the measurement that killed it. Recorded so the next
// person does not pay for the same experiments twice.
//
// 1. Mask fast path as an if/else inside the loop (tile entirely below the
//    diagonal => skip compare+select). Measured 15-24% SLOWER on all six shapes,
//    including dense, which takes the cheap path every single time. A branch
//    between the QK mma and the softmax splits the loop body into three basic
//    blocks and ptxas can no longer schedule across that boundary; at 34.6% issue
//    utilisation the lost overlap costs more than the ~96 ALU instructions saved.
//
// 2. The same idea done properly, by splitting the tile range into a no-mask loop
//    and a masked loop (both straight-line, no branch in the body). This does not
//    survive the compiler: a helper that large is not inlined even with
//    __forceinline__. The cubin carried 40 CALL and 668 LDL/STL instructions (V3:
//    zero of each), the register count went to the 255 ceiling, and capping it to
//    168 for three blocks spilled 1 KB per thread. Textual duplication of the loop
//    body would fix it, at the price of ~90 duplicated lines for a saving that
//    experiment 1 already suggests is not there.
//
// 3. Also rejected earlier: chunk-major shared layout (242 -> 217 address
//    instructions, no more), tileKV=32 at d=64 (20.8 vs 13.8 instructions per
//    HMMA), single-stage V, deferred PV (202/255 registers), one barrier per tile
//    (3-6% slower).
// =============================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "flash_attention.h"

#define blocksize 128
#define WARP 32
#define q_rows 64

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

namespace cuda_op_lab::flash_attention {

namespace {

extern __shared__ char dynamic_smem[];

constexpr int kWarpsPerBlock = blocksize / WARP;         // 4
constexpr int kRowsPerWarp   = q_rows / kWarpsPerBlock;  // 16 == MMA_M

__host__ __device__ constexpr int v3b_tile_kv(int head_dim) { return (head_dim <= 64) ? 64 : 32; }

// d=64 can afford a double-buffered V; d=128 goes single, which costs the
// loop-end barrier and saves 8 KB.
template <int head_dim>
__host__ __device__ constexpr int v3b_v_stages() { return (head_dim <= 64) ? 2 : 1; }

template <int head_dim>
__host__ __device__ constexpr int v3b_k_bytes() { return 2 * v3b_tile_kv(head_dim) * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

template <int head_dim>
__host__ __device__ constexpr int v3b_v_bytes() { return v3b_v_stages<head_dim>() * v3b_tile_kv(head_dim) * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

template <int head_dim>
__host__ __device__ constexpr int v3b_q_stage_bytes() { return q_rows * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

template <int head_dim>
__host__ __device__ constexpr int v3b_smem_bytes()
{
    constexpr int tail = (v3b_v_bytes<head_dim>() > v3b_q_stage_bytes<head_dim>())
                             ? v3b_v_bytes<head_dim>()
                             : v3b_q_stage_bytes<head_dim>();
    return v3b_k_bytes<head_dim>() + tail;
}

template <int head_dim>
struct FragCount
{
    static constexpr int kQ = head_dim / MMA_K;
    static constexpr int kS = v3b_tile_kv(head_dim) / MMA_N;
    static constexpr int kP = v3b_tile_kv(head_dim) / MMA_K;
    static constexpr int kO = head_dim / MMA_N;
};

// ---------------------------------------------------------------------------
// cp.async
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async_16_zfill(void* sharedptr, const void* baseptr, bool valid)
{
#if __CUDA_ARCH__ >= 800
    const unsigned shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(sharedptr));
    const int src_size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2; \n"
                 :: "r"(shared_addr), "l"(baseptr), "r"(src_size) : "memory");
#else
    *reinterpret_cast<uint4*>(sharedptr) = valid ? *reinterpret_cast<const uint4*>(baseptr) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

template <int head_dim>
__forceinline__ __device__ int swz(int row, int chunk)
{
    constexpr int k_chunks = head_dim / 8;
    return (chunk ^ (row & (k_chunks - 1))) * 8;
}

// ---------------------------------------------------------------------------
// PTX wrappers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&dst)[4], const void* src)
{
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(src));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&dst)[4], const void* src)
{
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(src));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi)
{
    uint32_t out;
#if __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(hi), "f"(lo));
#else
    const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
    out = *reinterpret_cast<const uint32_t*>(&packed);
#endif
    return out;
}

__forceinline__ __device__ int frag_row_lo(int lane) { return lane >> 2; }
__forceinline__ __device__ int frag_row_hi(int lane) { return (lane >> 2) + 8; }
__forceinline__ __device__ int frag_col_base(int lane) { return (lane & 3) << 1; }

__forceinline__ __device__ float quad_reduce_max(float value)
{
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
    return value;
}

__forceinline__ __device__ float quad_reduce_sum(float value)
{
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    value += __shfl_xor_sync(0xffffffffu, value, 2);
    return value;
}

template <int head_dim>
__forceinline__ __device__ const __nv_bfloat16* k_addr(const __nv_bfloat16* base, int n0, int k0, int lane)
{
    const int m = lane >> 3;
    const int n = n0 + (m >> 1) * 8 + (lane & 7);
    const int k = k0 + (m & 1) * 8;
    return base + n * head_dim + swz<head_dim>(n, k / 8);
}

template <int head_dim>
__forceinline__ __device__ const __nv_bfloat16* v_addr(const __nv_bfloat16* base, int k0, int n0, int lane)
{
    const int m = lane >> 3;
    const int k = k0 + (m & 1) * 8 + (lane & 7);
    const int n = n0 + (m >> 1) * 8;
    return base + k * head_dim + swz<head_dim>(k, n / 8);
}

template <int head_dim>
__forceinline__ __device__ const __nv_bfloat16* a_addr(const __nv_bfloat16* base, int row0, int k0, int lane)
{
    const int m = lane >> 3;
    const int row = row0 + (m & 1) * 8 + (lane & 7);
    const int k = k0 + (m >> 1) * 8;
    return base + row * head_dim + swz<head_dim>(row, k / 8);
}

template <int head_dim>
__device__ __forceinline__ void stage_tile(__nv_bfloat16* dst_base, const __nv_bfloat16* src_base,
                                           int src_row_origin, int row_limit, int rows)
{
    constexpr int kChunksPerRow = head_dim / 8;
    const int vecs = rows * kChunksPerRow;

    for (int v = threadIdx.x; v < vecs; v += blocksize)
    {
        const int shared_row = v / kChunksPerRow;
        const int chunk = v % kChunksPerRow;
        const int global_row = src_row_origin + shared_row;
        const bool valid = (global_row <= row_limit);
        const int safe_row = valid ? global_row : 0;

        cp_async_16_zfill(&dst_base[shared_row * head_dim + swz<head_dim>(shared_row, chunk)],
                          &src_base[static_cast<size_t>(safe_row) * head_dim + chunk * 8],
                          valid);
    }
}

// ---------------------------------------------------------------------------
// Kernel. The only difference from v3_tensorcore.cu is the launch_bounds below.
// ---------------------------------------------------------------------------

// Only d=128 is constrained. d=64 already fits three blocks with 165 registers,
// and asking ptxas for a floor there changes its allocation (150 registers) for no
// occupancy gain -- which would make d=64 stop being a control against V3.
template <int head_dim>
__host__ __device__ constexpr int v3b_min_blocks() { return (head_dim <= 64) ? 2 : 3; }

template <int head_dim>
__global__ void __launch_bounds__(blocksize, v3b_min_blocks<head_dim>())
attention_kernel_tensorcore_v3b(const __nv_bfloat16* __restrict__ Q,
                                const __nv_bfloat16* __restrict__ K,
                                const __nv_bfloat16* __restrict__ V,
                                __nv_bfloat16* __restrict__ O,
                                int seq_len, float scale, bool causal)
{
    using FC = FragCount<head_dim>;
    constexpr int kTileKV = v3b_tile_kv(head_dim);
    constexpr int kVStages = v3b_v_stages<head_dim>();

    static_assert(q_rows % kWarpsPerBlock == 0, "q_rows must split across warps");
    static_assert(kRowsPerWarp == MMA_M, "one warp owns exactly one mma M tile");
    static_assert(kTileKV % MMA_K == 0, "PV walks tileKV in MMA_K steps");
    static_assert(kTileKV % MMA_N == 0, "QK walks tileKV in MMA_N steps");
    static_assert(FC::kS == 2 * FC::kP, "S sub-blocks pair into PV A fragments");
    static_assert(FC::kO % 2 == 0, "PV pairs head_dim n8 blocks");

    // shared layout: [ K stage 0 | K stage 1 | V stages (= Q staging scratch) ]
    __nv_bfloat16* sharedK = reinterpret_cast<__nv_bfloat16*>(dynamic_smem);
    __nv_bfloat16* sharedV = sharedK + 2 * kTileKV * head_dim;
    __nv_bfloat16* sharedQstage = sharedV;

    const int warp_index = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int bh = blockIdx.y;

    const size_t layer_offset = static_cast<size_t>(bh) * seq_len * head_dim;
    const int q_block_origin = blockIdx.x * q_rows;
    const int warp_row_origin = q_block_origin + warp_index * kRowsPerWarp;

    const int q_max = min(q_block_origin + q_rows - 1, seq_len - 1);
    const int key_limit = causal ? q_max : (seq_len - 1);
    const int nums_loop = (key_limit + kTileKV) / kTileKV;

    float running_max[2] = {-FLT_MAX, -FLT_MAX};
    float running_sum[2] = {0.0f, 0.0f};

    uint32_t q_frag[FC::kQ][4];
    float o_acc[FC::kO][4] = {};
    float s_acc[FC::kS][4];
    uint32_t p_frag[FC::kP][4];

    const int row_lo = warp_row_origin + frag_row_lo(lane);
    const int row_hi = warp_row_origin + frag_row_hi(lane);
    const int limit_lo = causal ? min(row_lo, seq_len - 1) : (seq_len - 1);
    const int limit_hi = causal ? min(row_hi, seq_len - 1) : (seq_len - 1);
    const int col_base = frag_col_base(lane);

    // =======================================================================
    // Prologue
    // =======================================================================
    stage_tile<head_dim>(sharedQstage, Q + layer_offset, q_block_origin, seq_len - 1, q_rows);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    for (int kk = 0; kk < FC::kQ; ++kk)
        ldmatrix_x4(q_frag[kk], a_addr<head_dim>(sharedQstage, warp_index * kRowsPerWarp, kk * MMA_K, lane));

    __syncthreads();

    stage_tile<head_dim>(sharedK, K + layer_offset, 0, key_limit, kTileKV);
    cp_async_commit();

    // =======================================================================
    // Main loop
    // =======================================================================
    for (int iteration = 0; iteration < nums_loop; ++iteration)
    {
        const int k_stage = iteration & 1;
        const int v_stage = (kVStages == 2) ? (iteration & 1) : 0;
        const int kv_origin = iteration * kTileKV;

        stage_tile<head_dim>(sharedK + (k_stage ^ 1) * kTileKV * head_dim, K + layer_offset,
                             (iteration + 1) * kTileKV, key_limit, kTileKV);
        cp_async_commit();

        stage_tile<head_dim>(sharedV + v_stage * kTileKV * head_dim, V + layer_offset,
                             kv_origin, key_limit, kTileKV);
        cp_async_commit();

        cp_async_wait<2>();
        __syncthreads();

        for (int j = 0; j < FC::kS; ++j)
            for (int e = 0; e < 4; ++e) s_acc[j][e] = 0.0f;

        for (int kk = 0; kk < head_dim / MMA_K; ++kk)
        {
            for (int pp = 0; pp < FC::kS / 2; ++pp)
            {
                uint32_t kb[4];
                ldmatrix_x4(kb, k_addr<head_dim>(sharedK + k_stage * kTileKV * head_dim,
                                                 pp * 2 * MMA_N, kk * MMA_K, lane));
                const uint32_t b0[2] = {kb[0], kb[1]};
                const uint32_t b1[2] = {kb[2], kb[3]};
                mma_m16n8k16_bf16(s_acc[2 * pp], q_frag[kk], b0, s_acc[2 * pp]);
                mma_m16n8k16_bf16(s_acc[2 * pp + 1], q_frag[kk], b1, s_acc[2 * pp + 1]);
            }
        }

        for (int j = 0; j < FC::kS; ++j)
        {
            const int col = kv_origin + j * MMA_N + col_base;
            s_acc[j][0] *= scale;
            s_acc[j][1] *= scale;
            s_acc[j][2] *= scale;
            s_acc[j][3] *= scale;
            if (col     > limit_lo) s_acc[j][0] = -FLT_MAX;
            if (col + 1 > limit_lo) s_acc[j][1] = -FLT_MAX;
            if (col     > limit_hi) s_acc[j][2] = -FLT_MAX;
            if (col + 1 > limit_hi) s_acc[j][3] = -FLT_MAX;
        }

        for (int h = 0; h < 2; ++h)
        {
            const int e0 = h * 2;

            float local_max = -FLT_MAX;
            for (int j = 0; j < FC::kS; ++j)
                local_max = fmaxf(local_max, fmaxf(s_acc[j][e0], s_acc[j][e0 + 1]));

            const float previous_max = running_max[h];
            const float new_max = fmaxf(previous_max, quad_reduce_max(local_max));
            const float rescale = __expf(previous_max - new_max);
            running_max[h] = new_max;

            float local_sum = 0.0f;
            for (int j = 0; j < FC::kS; ++j)
            {
                const float a = __expf(s_acc[j][e0]     - new_max);
                const float b = __expf(s_acc[j][e0 + 1] - new_max);
                s_acc[j][e0] = a;
                s_acc[j][e0 + 1] = b;
                local_sum += a + b;
            }
            running_sum[h] = running_sum[h] * rescale + quad_reduce_sum(local_sum);

            for (int n = 0; n < FC::kO; ++n)
            {
                o_acc[n][e0]     *= rescale;
                o_acc[n][e0 + 1] *= rescale;
            }
        }

        for (int kk = 0; kk < FC::kP; ++kk)
        {
            p_frag[kk][0] = pack_bf16x2(s_acc[2 * kk][0], s_acc[2 * kk][1]);
            p_frag[kk][1] = pack_bf16x2(s_acc[2 * kk][2], s_acc[2 * kk][3]);
            p_frag[kk][2] = pack_bf16x2(s_acc[2 * kk + 1][0], s_acc[2 * kk + 1][1]);
            p_frag[kk][3] = pack_bf16x2(s_acc[2 * kk + 1][2], s_acc[2 * kk + 1][3]);
        }

        cp_async_wait<1>();
        __syncthreads();

        for (int kk = 0; kk < FC::kP; ++kk)
        {
            for (int pp = 0; pp < FC::kO / 2; ++pp)
            {
                uint32_t vb[4];
                ldmatrix_x4_trans(vb, v_addr<head_dim>(sharedV + v_stage * kTileKV * head_dim,
                                                       kk * MMA_K, pp * 2 * MMA_N, lane));
                const uint32_t b0[2] = {vb[0], vb[1]};
                const uint32_t b1[2] = {vb[2], vb[3]};
                mma_m16n8k16_bf16(o_acc[2 * pp], p_frag[kk], b0, o_acc[2 * pp]);
                mma_m16n8k16_bf16(o_acc[2 * pp + 1], p_frag[kk], b1, o_acc[2 * pp + 1]);
            }
        }

        if (kVStages == 1) __syncthreads();
    }

    // =======================================================================
    // Epilogue
    // =======================================================================
    const float inv_lo = running_sum[0] > 0.0f ? 1.0f / running_sum[0] : 0.0f;
    const float inv_hi = running_sum[1] > 0.0f ? 1.0f / running_sum[1] : 0.0f;

    __nv_bfloat16* out_lo = O + layer_offset + static_cast<size_t>(row_lo) * head_dim;
    __nv_bfloat16* out_hi = O + layer_offset + static_cast<size_t>(row_hi) * head_dim;

    for (int n = 0; n < FC::kO; ++n)
    {
        const int col = n * MMA_N + col_base;
        if (row_lo < seq_len)
            *reinterpret_cast<__nv_bfloat162*>(out_lo + col) =
                __floats2bfloat162_rn(o_acc[n][0] * inv_lo, o_acc[n][1] * inv_lo);
        if (row_hi < seq_len)
            *reinterpret_cast<__nv_bfloat162*>(out_hi + col) =
                __floats2bfloat162_rn(o_acc[n][2] * inv_hi, o_acc[n][3] * inv_hi);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

bool tensorcore_v3b_supported(int head_dim)
{
    return head_dim == 64 || head_dim == 128;
}

void launch_attention_tensorcore_v3b(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O, const AttentionShape& shape, bool causal, cudaStream_t stream)
{
    const dim3 grid(static_cast<unsigned>((shape.seq_len + q_rows - 1) / q_rows),
                    static_cast<unsigned>(shape.batch * shape.heads));
    const float scale = default_scale(shape.head_dim);

    switch (shape.head_dim)
    {
        case 64: {
            constexpr int smem = v3b_smem_bytes<64>();
            attention_kernel_tensorcore_v3b<64><<<grid, blocksize, smem, stream>>>(Q, K, V, O, shape.seq_len, scale, causal);
            break;
        }
        case 128: {
            constexpr int smem = v3b_smem_bytes<128>();
            cudaFuncSetAttribute(attention_kernel_tensorcore_v3b<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            attention_kernel_tensorcore_v3b<128><<<grid, blocksize, smem, stream>>>(Q, K, V, O, shape.seq_len, scale, causal);
            break;
        }
        default:
            return;
    }
}

FusedTilingInfo query_tensorcore_v3b(int head_dim)
{
    FusedTilingInfo info{};
    info.supported = tensorcore_v3b_supported(head_dim);
    if (!info.supported) return info;

    info.block_size = blocksize;
    info.tile_kv = v3b_tile_kv(head_dim);
    info.rows_per_block = q_rows;

    const int dynamic_bytes = (head_dim == 64) ? v3b_smem_bytes<64>() : v3b_smem_bytes<128>();
    const void* func = (head_dim == 64)
                           ? reinterpret_cast<const void*>(attention_kernel_tensorcore_v3b<64>)
                           : reinterpret_cast<const void*>(attention_kernel_tensorcore_v3b<128>);

    if (dynamic_bytes > 48 * 1024) cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);

    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(&attributes, func);

    info.shared_bytes_per_block = static_cast<int>(attributes.sharedSizeBytes) + dynamic_bytes;
    info.registers_per_thread = attributes.numRegs;
    info.local_bytes_per_thread = static_cast<int>(attributes.localSizeBytes);

    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, func, blocksize, dynamic_bytes);
    info.max_blocks_per_sm = blocks;

    int device = 0;
    cudaGetDevice(&device);
    int shared_per_sm = 0;
    cudaDeviceGetAttribute(&shared_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    info.device_shared_bytes_per_sm = shared_per_sm;
    info.shared_limited_blocks_per_sm = (info.shared_bytes_per_block > 0) ? shared_per_sm / info.shared_bytes_per_block : 0;

    return info;
}

double tensorcore_v3b_hbm_bytes(const AttentionShape& shape, bool causal)
{
    return tensorcore_v3_hbm_bytes(shape, causal);
}

}  // namespace cuda_op_lab::flash_attention
