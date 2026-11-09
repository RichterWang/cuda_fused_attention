// =============================================================================
// V3: Tensor Core (mma.m16n8k16, bf16) fused attention -- SKELETON
//
// Why q_rows goes 8 -> 64 together with mma (not as a separate optimisation):
//   intensity = (q_rows * tileKV * 4d) / (4 * tileKV * d) = q_rows  [FLOP/B]
//   q_rows=8  -> 8 FLOP/B  -> DRAM roof ~4.9 TFLOPS  (3070 Ti, 608 GB/s)
//   q_rows=64 -> 62 FLOP/B -> DRAM roof ~37.8 TFLOPS
// V2b already sits at ~70% of its own 8 FLOP/B roof, so swapping FMA for mma
// while keeping q_rows=8 caps the gain near 1.4x. m16n8k16 gives one warp 16
// query rows for free, so 4 warps/block makes q_rows=64 the natural tile.
//
// S -> P costs nothing: the C fragments of two adjacent n8 sub-blocks of S
// concatenate register-for-register into the A fragment of the next PV mma
// (first supplies k=0..7, second k=8..15). P never returns to shared memory,
// so V2b's score[] buffer and the two barriers around it are gone.
//
// Row reduction becomes a quad reduction: in the m16n8 C layout one row lives
// entirely inside one quad (equal lane/4), so online softmax needs 2 shuffles
// per row per tile instead of 5, with no block-level barrier in the phase.
//
// Shared memory (bf16): d=64 K double 16K + V double 16K = 32K -> 3 blk/SM
//                       d=128 K double 32K + V single 16K = 48K -> 2 blk/SM
// Q is not resident: it is staged into the V region, lifted to registers via
// ldmatrix, and the region is handed back to V. That is what buys 2 blk/SM
// at d=128.
//
// The swizzle key differs from V2b on purpose. V2b's (row>>1)&(k_chunks-1) is
// 2-way conflicting for ldmatrix, whose phase spans 8 consecutive rows where
// (row>>1) takes only 4 values. V3 uses row&(k_chunks-1).
//
// scale is applied to the f32 S accumulator, never to Q: Q must be bf16 in
// shared memory for ldmatrix, and pre-multiplying burns its 8-bit mantissa.
//
// Numerics: P must be bf16 for the second mma, ~2^-9 relative per element, so
// row-scaled error is expected around 3e-3..1e-2 at N=2048 (~2^-9 * sqrt(N)).
// Same trade-off upstream FlashAttention accepts (f32 accumulators, reduced
// precision P), but V3 carries its own documented tolerance rather than
// relaxing the shared one.
//
// Build-up order, each step separately verifiable:
//   1. single warp, q_rows=16, dense, d=64 -- validates fragment indexing and
//      the S->P hand-off and nothing else
//   2. 4 warps, q_rows=64
//   3. causal masking
//   4. d=128 (watch local_bytes_per_thread for spills)
// =============================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "flash_attention.h"

#define blocksize 128
#define WARP 32
#define tileKV 64
#define q_rows 64

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

namespace cuda_op_lab::flash_attention {

namespace {

extern __shared__ char dynamic_smem[];

// ---------------------------------------------------------------------------
// Tile geometry
// ---------------------------------------------------------------------------

constexpr int kWarpsPerBlock = blocksize / WARP;         // 4
constexpr int kRowsPerWarp   = q_rows / kWarpsPerBlock;  // 16 == MMA_M

static_assert(q_rows % kWarpsPerBlock == 0, "q_rows must split across warps");
static_assert(kRowsPerWarp == MMA_M, "one warp owns exactly one mma M tile");
static_assert(tileKV % MMA_K == 0, "PV walks tileKV in MMA_K steps");
static_assert(tileKV % MMA_N == 0, "QK walks tileKV in MMA_N steps");

// d=64 can afford a double-buffered V; d=128 goes single, matching the D-group
// ablation result (fewer shared bytes, 2 blocks/SM, ~18% lower latency).
template <int head_dim>
constexpr int v3_v_stages() { return (head_dim <= 64) ? 2 : 1; }

template <int head_dim>
constexpr int v3_k_bytes() { return 2 * tileKV * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

template <int head_dim>
constexpr int v3_v_bytes() { return v3_v_stages<head_dim>() * tileKV * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

template <int head_dim>
constexpr int v3_q_stage_bytes() { return q_rows * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

// The V region doubles as the Q staging scratch, so it must fit the larger.
template <int head_dim>
constexpr int v3_smem_bytes()
{
    constexpr int tail = (v3_v_bytes<head_dim>() > v3_q_stage_bytes<head_dim>())
                             ? v3_v_bytes<head_dim>() : v3_q_stage_bytes<head_dim>();
    return v3_k_bytes<head_dim>() + tail;
}
// d=64: 32768    d=128: 49152

// Fragments held per lane. Register estimate at d=128:
//   Q 8*4=32 | O 16*4=64 | S 8*4=32 | softmax 4 | addressing ~20 => ~152
// At d=64: ~84. If d=128 spills (local_bytes_per_thread > 0), fall back in
// this order: (a) softmax granularity 32 so S drops 32->16, (b) tileKV 32,
// (c) only then push O to shared, which mostly cancels the mma win. Do not
// reach for __launch_bounds__ first -- it just forces the spill.
template <int head_dim>
struct FragCount
{
    static constexpr int kQ = head_dim / MMA_K;  // A fragments of Q
    static constexpr int kS = tileKV / MMA_N;    // C fragments of S
    static constexpr int kP = tileKV / MMA_K;    // A fragments of P
    static constexpr int kO = head_dim / MMA_N;  // C fragments of O
};

// ---------------------------------------------------------------------------
// cp.async (carried over from V2b unchanged)
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

// ---------------------------------------------------------------------------
// Swizzle. Row stride is 128 B (d=64) or 256 B (d=128), both multiples of
// 128 B, so unswizzled the 8 rows of one ldmatrix phase collide in the same
// 4-bank group. XOR-ing the chunk index with the row index spreads rows 0..7
// over 8 distinct chunks in one aligned 128 B group: 32 banks, one hit each.
// ---------------------------------------------------------------------------

template <int head_dim>
__forceinline__ __device__ int swz(int row, int chunk)
{
    constexpr int k_chunks = head_dim / 8;
    return (chunk ^ (row & (k_chunks - 1))) * 8;
}

// ---------------------------------------------------------------------------
// PTX wrappers
// ---------------------------------------------------------------------------

// Non-trans: used for K, stored [kv][d] = [n][k], already the B layout of
// S = Q * K^T.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&dst)[4], const void* src)
{
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(src));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(addr));
}

// Trans: used for V, stored [kv][d] = [k][n] for O += P * V. V is deliberately
// not stored transposed -- it arrives row-major and a transposed store would
// forbid 16 B cp.async, forcing an element-wise scatter.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&dst)[4], const void* src)
{
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(src));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(addr));
}

// D = A*B + C.  A: m16k16 bf16 (4xb32), B: k16n8 bf16 (2xb32), C/D: f32 x4
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// The only lowering step between S and P. No lane-crossing movement.
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

// ---------------------------------------------------------------------------
// Fragment layouts, per lane.
//
// m16n8 C/D (f32 x4):
//   c[0],c[1] -> row = lane/4,     col = (lane%4)*2 + {0,1}
//   c[2],c[3] -> row = lane/4 + 8, col = (lane%4)*2 + {0,1}
//
// m16k16 A (bf16 x8 = 4 x b32):
//   a[0] -> row = lane/4,     k = (lane%4)*2 + {0,1}
//   a[1] -> row = lane/4 + 8, k = (lane%4)*2 + {0,1}
//   a[2] -> row = lane/4,     k = (lane%4)*2 + 8 + {0,1}
//   a[3] -> row = lane/4 + 8, k = (lane%4)*2 + 8 + {0,1}
//
// Two consequences the design leans on: a C row is confined to one quad, and
// S sub-blocks j, j+1 concatenate into the PV A fragment.
//
// TODO(step 1): confirm this mapping on hardware with the single-warp, d=64,
// dense build first. Every indexing mistake here reads as "wrong numbers, no
// crash", which is far more expensive to debug later.
// ---------------------------------------------------------------------------

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

// 16 B cp.async per lane into a swizzled destination. Shared with Q, K and V
// staging since all three are [row][head_dim] row-major in global memory.
template <int head_dim>
__device__ __forceinline__ void stage_tile(__nv_bfloat16* dst_base, const __nv_bfloat16* src_base, int src_row_origin, int row_limit, int rows)
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
// Kernel
// ---------------------------------------------------------------------------

template <int head_dim>
void __global__ attention_kernel_tensorcore_v3(const __nv_bfloat16* __restrict__ Q,
                                               const __nv_bfloat16* __restrict__ K,
                                               const __nv_bfloat16* __restrict__ V,
                                               __nv_bfloat16* __restrict__ O,
                                               int seq_len, float scale, bool causal)
{
    using FC = FragCount<head_dim>;
    constexpr int kVStages = v3_v_stages<head_dim>();

    // shared layout: [ K stage 0 | K stage 1 | V stages (= Q staging scratch) ]
    __nv_bfloat16* sharedK = reinterpret_cast<__nv_bfloat16*>(dynamic_smem);
    __nv_bfloat16* sharedV = sharedK + 2 * tileKV * head_dim;
    __nv_bfloat16* sharedQstage = sharedV;  // reclaimed by V after the prologue

    const int warp_index = threadIdx.x / WARP;
    const int lane_index = threadIdx.x % WARP;
    const int bh = blockIdx.y;

    const size_t layer_offset = static_cast<size_t>(bh) * seq_len * head_dim;
    const int q_block_origin = blockIdx.x * q_rows;
    const int warp_row_origin = q_block_origin + warp_index * kRowsPerWarp;

    const int q_max = q_block_origin + q_rows - 1;
    const int key_limit = causal ? min(q_max, seq_len - 1) : (seq_len - 1);
    const int nums_loop = (key_limit + tileKV) / tileKV;

    // Each lane owns two softmax rows: frag_row_lo and frag_row_hi.
    float running_max[2] = {-FLT_MAX, -FLT_MAX};
    float running_sum[2] = {0.0f, 0.0f};

    uint32_t q_frag[FC::kQ][4];   // Q lives in registers for the whole kernel
    float o_acc[FC::kO][4] = {};  // f32, never lowered mid-flight
    float s_acc[FC::kS][4];       // QK output, reused as the softmax workspace
    uint32_t p_frag[FC::kP][4];   // S reinterpreted as the PV A operand

    // =======================================================================
    // Prologue: stage Q into the V scratch, lift to registers, hand the
    // scratch back to V, then prime the K pipeline two tiles deep.
    // =======================================================================
    // TODO(step 1):
    //   stage_tile<head_dim>(sharedQstage, Q + layer_offset, q_block_origin, seq_len - 1, q_rows);
    //   cp_async_commit(); cp_async_wait<0>(); __syncthreads();
    //   for (kk = 0; kk < FC::kQ; ++kk)
    //       ldmatrix_x4(q_frag[kk], &sharedQstage[<warp rows, k chunk kk, swizzled>]);
    //   __syncthreads();                       // scratch belongs to V again
    //   stage_tile<head_dim>(sharedK, K + layer_offset, 0, key_limit, tileKV);
    //   cp_async_commit();
    //   stage_tile<head_dim>(sharedK + tileKV * head_dim, K + layer_offset, tileKV, key_limit, tileKV);
    //   cp_async_commit();

    // =======================================================================
    // Main loop over KV tiles.
    // =======================================================================
    for (int iteration = 0; iteration < nums_loop; ++iteration)
    {
        const int k_stage = iteration & 1;
        const int v_stage = (kVStages == 2) ? (iteration & 1) : 0;
        const int kv_origin = iteration * tileKV;

        // -- stage V for this tile, K for tile+2 ---------------------------
        // TODO: stage_tile V -> sharedV + v_stage * tileKV * head_dim; commit
        //       if (iteration + 2 < nums_loop) stage_tile K -> sharedK + k_stage; commit
        //       Keep the committed-group count and the wait depth in lockstep;
        //       an off-by-one here silently reads a half-written tile.

        // -- QK^T: S[j] = Q * K^T over tileKV/MMA_N sub-blocks -------------
        // TODO: cp_async_wait<N_k>(); __syncthreads();
        //       zero s_acc
        //       for (kk = 0; kk < head_dim/MMA_K; ++kk) {
        //           ldmatrix_x4(kb, &sharedK[k_stage ... kk ...]);   // 4 n8 blocks
        //           for (j = 0; j < FC::kS; ++j)
        //               mma_m16n8k16_bf16(s_acc[j], q_frag[kk], kb_pair(j), s_acc[j]);
        //       }
        //       d=128, tileKV=64: 8*8 = 64 mma per warp per tile.

        // -- scale + mask ---------------------------------------------------
        // TODO: s_acc[j][e] *= scale  (scale is applied here, not on Q)
        //       Then mask. cp.async zfill makes out-of-range K read as 0, and
        //       S=0 is not -inf, so the tail tile must be masked explicitly.
        //       Each lane derives its own (row, col) from frag_row_lo/hi and
        //       frag_col_base, which is simpler than V2b's version:
        //         row = warp_row_origin + frag_row_{lo,hi}(lane_index)
        //         col = kv_origin + j * MMA_N + frag_col_base(lane_index) + {0,1}
        //         mask if col > (causal ? min(row, seq_len-1) : seq_len-1)

        // -- online softmax, no __syncthreads ------------------------------
        // TODO: for h in {lo, hi}:
        //         m_local = max over own 2 cols x FC::kS sub-blocks
        //         m_new   = max(running_max[h], quad_reduce_max(m_local))
        //         rescale = __expf(running_max[h] - m_new); running_max[h] = m_new
        //         s_acc -> __expf(s - m_new) in place
        //         running_sum[h] = running_sum[h] * rescale + quad_reduce_sum(local_sum)
        //         o_acc[n][{0,1}] *= rescale_lo;  o_acc[n][{2,3}] *= rescale_hi
        //       2 shuffles per row per tile (V2b: 5), no shared scratch.

        // -- S -> P, zero movement -----------------------------------------
        // TODO: for (kk = 0; kk < FC::kP; ++kk) {
        //           j = 2*kk;    // sub-block j -> k 0..7, j+1 -> k 8..15
        //           p_frag[kk][0] = pack_bf16x2(s_acc[j  ][0], s_acc[j  ][1]);
        //           p_frag[kk][1] = pack_bf16x2(s_acc[j  ][2], s_acc[j  ][3]);
        //           p_frag[kk][2] = pack_bf16x2(s_acc[j+1][0], s_acc[j+1][1]);
        //           p_frag[kk][3] = pack_bf16x2(s_acc[j+1][2], s_acc[j+1][3]);
        //       }
        //       Verify this pairing against the A-layout comment above in
        //       step 1; it is the single highest-value assertion in the file.

        // -- PV: O += P * V over d/MMA_N sub-blocks ------------------------
        // TODO: cp_async_wait<N_v>(); __syncthreads();
        //       for (kk = 0; kk < FC::kP; ++kk) {
        //           ldmatrix_x4_trans(vb, &sharedV[v_stage ... kk ...]);
        //           for (n = 0; n < FC::kO; ++n)
        //               mma_m16n8k16_bf16(o_acc[n], p_frag[kk], vb_pair(n), o_acc[n]);
        //       }
        //       d=128, tileKV=64: 16*4 = 64 mma per warp per tile.

        (void)k_stage; (void)v_stage; (void)kv_origin;
    }

    // =======================================================================
    // Epilogue: normalise and write back in the C layout.
    // =======================================================================
    // TODO: const float inv_lo = running_sum[0] > 0.0f ? 1.0f / running_sum[0] : 0.0f;
    //       same for hi. V2b divides unguarded, which yields inf when a whole
    //       row is masked; V3 restores the V1 guard.
    //       Then, for n in [0, FC::kO):
    //         col = n * MMA_N + frag_col_base(lane_index)
    //         row_lo = warp_row_origin + frag_row_lo(lane_index) -> o_acc[n][0..1] * inv_lo
    //         row_hi = warp_row_origin + frag_row_hi(lane_index) -> o_acc[n][2..3] * inv_hi
    //       Two adjacent cols per lane => one __nv_bfloat162 store each.

    (void)Q; (void)K; (void)V; (void)O; (void)scale;
    (void)sharedQstage; (void)warp_row_origin; (void)lane_index; (void)layer_offset;
    (void)q_frag; (void)o_acc; (void)s_acc; (void)p_frag;
    (void)running_max; (void)running_sum;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

bool tensorcore_v3_supported(int head_dim)
{
    return head_dim == 64 || head_dim == 128;
}

void launch_attention_tensorcore_v3(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O, const AttentionShape& shape, bool causal, cudaStream_t stream)
{
    const dim3 grid(static_cast<unsigned>((shape.seq_len + q_rows - 1) / q_rows),
                    static_cast<unsigned>(shape.batch * shape.heads));
    const float scale = default_scale(shape.head_dim);

    switch (shape.head_dim)
    {
        case 64: {
            constexpr int smem = v3_smem_bytes<64>();
            attention_kernel_tensorcore_v3<64><<<grid, blocksize, smem, stream>>>(Q, K, V, O, shape.seq_len, scale, causal);
            break;
        }
        case 128: {
            constexpr int smem = v3_smem_bytes<128>();
            // 48 KB exactly, so no opt-in is strictly required, but keep the
            // call so a later tileKV bump does not silently fail to launch.
            cudaFuncSetAttribute(attention_kernel_tensorcore_v3<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            attention_kernel_tensorcore_v3<128><<<grid, blocksize, smem, stream>>>(Q, K, V, O, shape.seq_len, scale, causal);
            break;
        }
        default:
            return;
    }
}

FusedTilingInfo query_tensorcore_v3(int head_dim)
{
    FusedTilingInfo info{};
    info.supported = tensorcore_v3_supported(head_dim);
    if (!info.supported) return info;

    info.block_size = blocksize;
    info.tile_kv = tileKV;
    info.rows_per_block = q_rows;

    const int dynamic_bytes = (head_dim == 64) ? v3_smem_bytes<64>() : v3_smem_bytes<128>();
    const void* func = (head_dim == 64)
                           ? reinterpret_cast<const void*>(attention_kernel_tensorcore_v3<64>)
                           : reinterpret_cast<const void*>(attention_kernel_tensorcore_v3<128>);

    if (dynamic_bytes > 48 * 1024) cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);

    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(&attributes, func);

    info.shared_bytes_per_block = static_cast<int>(attributes.sharedSizeBytes) + dynamic_bytes;
    info.registers_per_thread = attributes.numRegs;
    // Watch this one at d=128: the estimate is ~152 registers/lane, close
    // enough to the 255 ceiling that the compiler may spill for scheduling.
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

// Algorithmic DRAM traffic. Same accounting as V2b, but with q_rows=64 the
// K/V re-read factor drops 8x, which is the whole point of the version.
double tensorcore_v3_hbm_bytes(const AttentionShape& shape, bool causal)
{
    const double element = static_cast<double>(sizeof(__nv_bfloat16));
    const int slabs = shape.batch * shape.heads;
    const int row_blocks = (shape.seq_len + q_rows - 1) / q_rows;

    // Q read once, O written once.
    double bytes = 2.0 * slabs * shape.seq_len * shape.head_dim * element;

    // K/V tiles, counted per block that actually loads them (whole tiles, so
    // this matches the kernel's loop trip count exactly).
    double kv_tiles = 0.0;
    for (int block = 0; block < row_blocks; ++block)
    {
        const int q_max = block * q_rows + q_rows - 1;
        const int key_limit = causal ? ((q_max < shape.seq_len - 1) ? q_max : shape.seq_len - 1) : (shape.seq_len - 1);
        kv_tiles += static_cast<double>((key_limit + tileKV) / tileKV);
    }

    bytes += 2.0 * slabs * kv_tiles * tileKV * shape.head_dim * element;

    return bytes;
}

}  // namespace cuda_op_lab::flash_attention
