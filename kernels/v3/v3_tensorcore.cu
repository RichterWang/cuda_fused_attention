// =============================================================================
// V3: Tensor Core (mma.m16n8k16, bf16) fused attention -- dense and causal
//
// ---------------------------------------------------------------------------
// What it measures (RTX 3070 Ti Laptop, sm_86, 448 GB/s, ~1.41 GHz)
//
//   6 lite shapes        20-29 TFLOPS, 5.8-7.7x over V2b, 55-83x over V0
//   (2,8,2048,64) dense  0.619 ms, 27.7 TFLOPS
//
// Roofline, read carefully -- three traps live here:
//
//   * The algorithmic model (tensorcore_v3_hbm_bytes) gives 0.277 GB for that
//     shape, i.e. 448 GB/s at 0.619 ms. NCU measured 12.65 MB read + 4.10 MB
//     written instead: Q, K and V each read exactly once and O written exactly
//     once, the theoretical minimum, at a 95.5% L2 hit rate. One slab's K/V
//     (512 KB here) stays resident, so the model overestimates DRAM by 16x and
//     dram__throughput is 5.8%. This kernel is NOT bandwidth bound, and every
//     lite shape fits a slab's K/V in the 4 MB L2.
//   * Intensity is pinned at q_rows = 64 FLOP/B, so "TFLOPS" and "algorithmic
//     GB/s" are the same number in different units. 64 FLOP/B x 448 GB/s is
//     28.7 TFLOPS and we measure 27.7 -- that agreement is an artefact of the
//     constant ratio, not evidence that a DRAM roof binds. Do not report it.
//   * V2b's 8 FLOP/B roof at 448 GB/s is 3.58 TFLOPS and V2b measures 3.6, so
//     the older kernel really does sit on its algorithmic roof. The 7x gain is
//     therefore mostly instruction efficiency -- one mma replaces 64 lane FMAs
//     and the 5-shuffle row reduction disappears -- not saved bytes.
//
// Where the remaining time actually goes, same launch:
//   sm__inst_executed_pipe_tensor.sum               4,194,304
//     = 512 blocks x 32 tiles x 4 warps x 64 mma, exactly the designed count
//   sm__pipe_tensor_op_hmma_cycles_active           42.9% of peak
//   sm__warps_active                                23.3% (the cap is 25%)
//   shared bank conflicts                           18,679 of ~8.4M ldmatrix
//                                                   wavefronts = 0.2%
// 43% tensor occupancy with 23% warps and 5.8% DRAM means this is latency
// bound, not throughput bound: a tile is a strict QK mma -> mask -> exp -> pack
// -> PV mma sequence with two barriers and two cp.async waits inside it, and 12
// warps/SM cannot cover that chain. At ~18 cycles per mma per SMSP the hardware
// ceiling works out near 61 TFLOPS, so there is roughly 2.3x on the table. The
// lever is cross-tile software pipelining (softmax of tile i under the QK mma
// of tile i+1), which needs a second set of S fragments -- budget registers for
// it before trying, because d=128 already sits at 218.
//
// ---------------------------------------------------------------------------
// Design
//
// Tile shape. One warp owns exactly one m16 tile, so four warps give q_rows=64
// per block; that is why q_rows moves 8 -> 64 together with mma.
//
//   d=64 : tileKV=64 -> K double 16 KB + V double 16 KB                    = 32 KB
//   d=128: tileKV=32 -> K double 16 KB + V single 8 KB, Q scratch 16 KB    = 32 KB
//
// 32 KB either way, so shared memory alone allows 3 blocks/SM. Occupancy is
// decided by registers instead: ptxas gives 165 registers at d=64 (3 blocks,
// 12 warps) and 218 at d=128 (2 blocks, 8 warps). Capping to 170 compiles to
// 168 with zero spill, so __launch_bounds__(blocksize, 3) would buy the third
// block at d=128 -- measured but deliberately not applied yet, since only the
// GPU can say whether the tighter scheduling wins. Q is not resident: it is
// staged into the V region, lifted to registers with ldmatrix, and the region
// goes back to V.
//
// Row reduction. In the m16n8 C layout one row is confined to one quad (the four
// lanes with equal lane/4) and a lane holds 2 columns per n8 fragment, so one
// row of a 64-column tile is 4 lanes x 16 values. Online softmax therefore costs
// 2 shuffles per row, with no shared scratch and no __syncthreads in the phase.
// The reduced value is replicated inside the quad, so each lane rescales its own
// O columns directly: C rows are lo/hi in c[0..1] / c[2..3] of every fragment.
//
// S -> P costs nothing: the C fragments of two adjacent n8 sub-blocks of S
// concatenate register-for-register into the A fragment of the next PV mma
// (j supplies k=0..7, j+1 supplies k=8..15). P never returns to shared memory.
//
// Pipeline. K is double buffered and prefetched one tile ahead: iteration i
// reads buffer i&1 and fills (i+1)&1, so the buffer written is never the one
// read. Every iteration commits exactly two groups (K next, then V now) and
// waits 2 then 1, which keeps the group count and the wait depth in lockstep;
// the last K prefetch is still issued because dropping it would change that
// count. Rows past key_limit are zero-filled by cp.async, and S=0 is masked to
// -FLT_MAX explicitly (zfill is 0, not -inf). V is single stage at d=128, so
// only that build needs the loop-end barrier.
//
// Swizzle. The key differs from V2b on purpose: V2b's (row>>1)&(k_chunks-1) is
// 2-way conflicting for ldmatrix, whose phase spans 8 consecutive rows where
// (row>>1) takes only 4 values. V3 uses row&(k_chunks-1). cp.async and ldmatrix
// both move exactly one 16 B chunk, so staging and reading agree.
//
// Numerics. P is bf16 for the second mma (~2^-9 relative per element). That
// shows up as a much larger per-element relative error than V2 (max |d|/|ref|
// reaches 3e2 on near-zero output elements) while the row-scaled error stays at
// 1.00 bf16 ULP, which is exactly why the project gates on the row-scaled metric.
// Measured 1.00 ULP on every accuracy shape (N=1..256, d=64/128, dense/causal),
// so this kernel is held to the same 2.0 gate as the others -- no separate
// tolerance. scale is applied to the f32 S accumulator, never to Q: Q must stay
// bf16 in shared memory for ldmatrix.
//
// Masking. -FLT_MAX, not -INFINITY: a fully masked row would otherwise compute
// exp(-inf - -inf) = NaN. Once a row has had a valid column its running_max is
// finite, so a later fully masked tile contributes exp(-huge)=0 and the running
// sum is untouched. Rows past seq_len are never stored.
//
// Debug. Configure with -DFA_V3_DEBUG_DUMP=ON: block (0,0) writes its source Q
// rows, its first K/V tile, the raw S (before scale and mask) and the packed P
// into a __device__ buffer, and the launcher dumps them once per process as
// v3_dump_{q,k,s,p}.bin (fp32, row major). S can then be checked against a plain
// CPU Q.K^T without reproducing the benchmark's RNG: that separates a
// fragment-index bug from a softmax bug, which is the expensive distinction.
//
// Build-up order, each step separately verifiable. Steps 0-4 all pass:
//   0. single warp, q_rows=16, dense, d=64, N=64 -- compare dumped S against a
//      CPU dot product, nothing else
//   1. the same launch through the full pipeline -- S->P hand-off and O indexing
//   2. four warps, q_rows=64 (all six benchmark shapes)
//   3. causal masking (N=1 takes the fully masked tail, and is exact)
//   4. d=128 (tileKV=32, 218 registers, zero spill)
//   5. still open: cross-tile pipelining, then re-measure tensor occupancy
// =============================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>

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

// KV tile width. d=64 keeps 64 rows; d=128 drops to 32 so the tile is 16 KB
// there too and the register count follows. Both land on 32 KB per block.
__host__ __device__ constexpr int v3_tile_kv(int head_dim) { return (head_dim <= 64) ? 64 : 32; }

// d=64 can afford a double-buffered V; d=128 goes single, which costs the
// loop-end barrier and saves 8 KB.
template <int head_dim>
__host__ __device__ constexpr int v3_v_stages() { return (head_dim <= 64) ? 2 : 1; }

template <int head_dim>
__host__ __device__ constexpr int v3_k_bytes() { return 2 * v3_tile_kv(head_dim) * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

template <int head_dim>
__host__ __device__ constexpr int v3_v_bytes() { return v3_v_stages<head_dim>() * v3_tile_kv(head_dim) * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

template <int head_dim>
__host__ __device__ constexpr int v3_q_stage_bytes() { return q_rows * head_dim * static_cast<int>(sizeof(__nv_bfloat16)); }

// The V region doubles as the Q staging scratch, so it must fit the larger.
template <int head_dim>
__host__ __device__ constexpr int v3_smem_bytes()
{
    constexpr int tail = (v3_v_bytes<head_dim>() > v3_q_stage_bytes<head_dim>())
                             ? v3_v_bytes<head_dim>()
                             : v3_q_stage_bytes<head_dim>();
    return v3_k_bytes<head_dim>() + tail;
}
// d=64: 32768    d=128: 32768

// Fragments held per lane, d=128 with tileKV=32:
//   Q 8*4=32 | O 16*4=64 | S 4*4=16 | P 2*4=8 | softmax 4 | addressing ~20
//   => ~144, which still allows 3 blocks/SM (65536 / (128*3) = 170 per thread).
// d=64 with tileKV=64: 16+32+32+16+4+20 = ~120. If d=128 spills
// (local_bytes_per_thread > 0), drop tileKV to 16 first, then keep O's last
// column group in shared. Do not reach for __launch_bounds__ first -- it only
// forces the spill.
template <int head_dim>
struct FragCount
{
    static constexpr int kQ = head_dim / MMA_K;                    // A fragments of Q
    static constexpr int kS = v3_tile_kv(head_dim) / MMA_N;        // C fragments of S
    static constexpr int kP = v3_tile_kv(head_dim) / MMA_K;        // A fragments of P
    static constexpr int kO = head_dim / MMA_N;                    // C fragments of O
};

#ifdef V3_DEBUG_DUMP
// Four fp32 regions with fixed strides, so one reader handles both head_dims:
//   S: 64 rows x 64 cols (stride 64)   -- QK output, before scale and mask
//   P: 64 rows x 64 cols (stride 64)   -- packed bf16 P, widened back to f32
//   Q: 64 rows x head_dim (stride 128)
//   K: 64 rows x head_dim (stride 128) -- the first K/V tile
// Q and K are dumped so the checker needs nothing but these files.
constexpr int kDebugStride = 64;
constexpr int kDebugFeatureStride = 128;
constexpr int kDebugSOffset = 0;
constexpr int kDebugPOffset = q_rows * kDebugStride;
constexpr int kDebugQOffset = kDebugPOffset + q_rows * kDebugStride;
constexpr int kDebugKOffset = kDebugQOffset + q_rows * kDebugFeatureStride;
constexpr int kDebugFloats = kDebugKOffset + q_rows * kDebugFeatureStride;
__device__ float g_v3_debug[kDebugFloats];
#endif

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

// ---------------------------------------------------------------------------
// Swizzle. Row stride is 128 B (d=64) or 256 B (d=128), both multiples of 128 B,
// so unswizzled the 8 rows of one ldmatrix phase collide in the same 4-bank
// group. XOR-ing the 16 B chunk index with the row index spreads rows 0..7 over
// 8 distinct chunks: 32 banks, one hit each.
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

// Non-trans: used for Q and K. K is stored [kv][d] = [n][k], which is already
// the B layout of S = Q * K^T.
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

#ifdef V3_DEBUG_DUMP
__device__ __forceinline__ void unpack_bf16x2(uint32_t packed, float (&out)[2])
{
    const __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&packed);
    out[0] = __bfloat162float(value.x);
    out[1] = __bfloat162float(value.y);
}
#endif

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
// m16n8k16 B (bf16 x4 = 2 x b32):
//   b[0] -> n = lane/4, k = (lane%4)*2 + {0,1}
//   b[1] -> n = lane/4, k = (lane%4)*2 + 8 + {0,1}
//
// Two consequences the design leans on: a C row is confined to one quad, and
// S sub-blocks j, j+1 concatenate into the PV A fragment.
//
// ldmatrix.x4 gives one 8x8 tile per register, chosen by m = lane/8, and the
// group order differs between A and B -- this is the single most error-prone
// table in the file:
//   A (Q):  register index -> row group = m&1, k group = m>>1
//   B (K,V): register index -> k group  = m&1, n group = m>>1
// and the B order is what makes (d[0],d[1]) one n8 block and (d[2],d[3]) the
// next one, which is also the pairing S->P needs.
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

// Each lane points at one 16 B chunk: the k group is m&1 and the n group m>>1,
// so (d[0],d[1]) is one n8 block and (d[2],d[3]) is the next one.
template <int head_dim>
__forceinline__ __device__ const __nv_bfloat16* k_addr(const __nv_bfloat16* base, int n0, int k0, int lane)
{
    const int m = lane >> 3;
    const int n = n0 + (m >> 1) * 8 + (lane & 7);
    const int k = k0 + (m & 1) * 8;
    return base + n * head_dim + swz<head_dim>(n, k / 8);
}

// V is stored [kv][d] = [k][n], so the shared row is k and the chunk is n.
template <int head_dim>
__forceinline__ __device__ const __nv_bfloat16* v_addr(const __nv_bfloat16* base, int k0, int n0, int lane)
{
    const int m = lane >> 3;
    const int k = k0 + (m & 1) * 8 + (lane & 7);
    const int n = n0 + (m >> 1) * 8;
    return base + k * head_dim + swz<head_dim>(k, n / 8);
}

// Q and P are A fragments: the row group is m&1 and the k group m>>1.
template <int head_dim>
__forceinline__ __device__ const __nv_bfloat16* a_addr(const __nv_bfloat16* base, int row0, int k0, int lane)
{
    const int m = lane >> 3;
    const int row = row0 + (m & 1) * 8 + (lane & 7);
    const int k = k0 + (m >> 1) * 8;
    return base + row * head_dim + swz<head_dim>(row, k / 8);
}

// 16 B cp.async per lane into a swizzled destination. Shared with Q, K and V
// staging since all three are [row][head_dim] row-major in global memory.
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

#ifdef V3_DEBUG_DUMP
// Block (0,0) only, and only the first K/V tile, so the dump is one 64-row query
// block x one KV tile with block-local columns.
__device__ __forceinline__ void debug_store(int offset, int row, int col, int stride, float value)
{
    if (blockIdx.x == 0 && blockIdx.y == 0 && row < q_rows && col < stride)
        g_v3_debug[offset + row * stride + col] = value;
}
#endif

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int head_dim>
__global__ void attention_kernel_tensorcore_v3(const __nv_bfloat16* __restrict__ Q,
                                               const __nv_bfloat16* __restrict__ K,
                                               const __nv_bfloat16* __restrict__ V,
                                               __nv_bfloat16* __restrict__ O,
                                               int seq_len, float scale, bool causal)
{
    using FC = FragCount<head_dim>;
    constexpr int kTileKV = v3_tile_kv(head_dim);
    constexpr int kVStages = v3_v_stages<head_dim>();

    static_assert(q_rows % kWarpsPerBlock == 0, "q_rows must split across warps");
    static_assert(kRowsPerWarp == MMA_M, "one warp owns exactly one mma M tile");
    static_assert(kTileKV % MMA_K == 0, "PV walks tileKV in MMA_K steps");
    static_assert(kTileKV % MMA_N == 0, "QK walks tileKV in MMA_N steps");
    static_assert(FC::kS == 2 * FC::kP, "S sub-blocks pair into PV A fragments");
    static_assert(FC::kO % 2 == 0, "PV pairs head_dim n8 blocks");

    // shared layout: [ K stage 0 | K stage 1 | V stages (= Q staging scratch) ]
    __nv_bfloat16* sharedK = reinterpret_cast<__nv_bfloat16*>(dynamic_smem);
    __nv_bfloat16* sharedV = sharedK + 2 * kTileKV * head_dim;
    __nv_bfloat16* sharedQstage = sharedV;  // reclaimed by V after the prologue

    const int warp_index = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int bh = blockIdx.y;

    const size_t layer_offset = static_cast<size_t>(bh) * seq_len * head_dim;
    const int q_block_origin = blockIdx.x * q_rows;
    const int warp_row_origin = q_block_origin + warp_index * kRowsPerWarp;

    const int q_max = min(q_block_origin + q_rows - 1, seq_len - 1);
    const int key_limit = causal ? q_max : (seq_len - 1);
    const int nums_loop = (key_limit + kTileKV) / kTileKV;

#ifdef V3_DEBUG_DUMP
    // Straight from global memory, so the dump does not depend on the staging or
    // the swizzle. Idempotent, so every launch of block (0,0) writes the same
    // values; the host side only samples the first launch.
    for (int index = threadIdx.x; index < q_rows * head_dim; index += blocksize)
        debug_store(kDebugQOffset, index / head_dim, index % head_dim, kDebugFeatureStride,
                    __bfloat162float(Q[layer_offset + index]));
    for (int index = threadIdx.x; index < kTileKV * head_dim; index += blocksize)
        debug_store(kDebugKOffset, index / head_dim, index % head_dim, kDebugFeatureStride,
                    __bfloat162float(K[layer_offset + index]));
#endif

    // Each lane owns two softmax rows: frag_row_lo and frag_row_hi.
    float running_max[2] = {-FLT_MAX, -FLT_MAX};
    float running_sum[2] = {0.0f, 0.0f};

    uint32_t q_frag[FC::kQ][4];   // Q lives in registers for the whole kernel
    float o_acc[FC::kO][4] = {};  // f32, never lowered mid-flight
    float s_acc[FC::kS][4];       // QK output, reused as the softmax workspace
    uint32_t p_frag[FC::kP][4];   // S reinterpreted as the PV A operand

    // =======================================================================
    // Prologue: stage Q into the V scratch, lift it to registers, hand the
    // scratch back to V, then prime the K pipeline one tile ahead.
    // =======================================================================
    stage_tile<head_dim>(sharedQstage, Q + layer_offset, q_block_origin, seq_len - 1, q_rows);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    for (int kk = 0; kk < FC::kQ; ++kk)
        ldmatrix_x4(q_frag[kk], a_addr<head_dim>(sharedQstage, warp_index * kRowsPerWarp, kk * MMA_K, lane));

    __syncthreads();  // the scratch belongs to V again

    stage_tile<head_dim>(sharedK, K + layer_offset, 0, key_limit, kTileKV);
    cp_async_commit();

    // =======================================================================
    // Main loop over KV tiles.
    // =======================================================================
    for (int iteration = 0; iteration < nums_loop; ++iteration)
    {
        const int k_stage = iteration & 1;
        const int v_stage = (kVStages == 2) ? (iteration & 1) : 0;
        const int kv_origin = iteration * kTileKV;

        // -- prefetch next K (other buffer) and stage this V ----------------
        // Always commit, even for the tile past the end: the wait depths below
        // assume exactly two groups per iteration.
        stage_tile<head_dim>(sharedK + (k_stage ^ 1) * kTileKV * head_dim, K + layer_offset,
                             (iteration + 1) * kTileKV, key_limit, kTileKV);
        cp_async_commit();

        stage_tile<head_dim>(sharedV + v_stage * kTileKV * head_dim, V + layer_offset,
                             kv_origin, key_limit, kTileKV);
        cp_async_commit();

        // Pending: [K_i, K_i+1, V_i]. Leaving two outstanding completes K_i.
        cp_async_wait<2>();
        __syncthreads();

        // -- QK^T: S[j] = Q * K^T over the tile ----------------------------
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

        const int row_lo = warp_row_origin + frag_row_lo(lane);
        const int row_hi = warp_row_origin + frag_row_hi(lane);

#ifdef V3_DEBUG_DUMP
        if (iteration == 0)
        {
            for (int j = 0; j < FC::kS; ++j)
            {
                const int col = j * MMA_N + frag_col_base(lane);
                debug_store(kDebugSOffset, row_lo, col, kDebugStride, s_acc[j][0]);
                debug_store(kDebugSOffset, row_lo, col + 1, kDebugStride, s_acc[j][1]);
                debug_store(kDebugSOffset, row_hi, col, kDebugStride, s_acc[j][2]);
                debug_store(kDebugSOffset, row_hi, col + 1, kDebugStride, s_acc[j][3]);
            }
        }
#endif

        // -- scale + mask ---------------------------------------------------
        // zfill gives 0, not -inf, so the tail tile must be masked explicitly.
        // -FLT_MAX keeps exp(-inf - -inf) = NaN out of reach.
        const int limit_lo = causal ? min(row_lo, seq_len - 1) : (seq_len - 1);
        const int limit_hi = causal ? min(row_hi, seq_len - 1) : (seq_len - 1);
        const int col_base = frag_col_base(lane);

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

        // -- online softmax: two quad reductions per row, no barrier --------
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

        // -- S -> P, zero movement ------------------------------------------
        for (int kk = 0; kk < FC::kP; ++kk)
        {
            p_frag[kk][0] = pack_bf16x2(s_acc[2 * kk][0], s_acc[2 * kk][1]);
            p_frag[kk][1] = pack_bf16x2(s_acc[2 * kk][2], s_acc[2 * kk][3]);
            p_frag[kk][2] = pack_bf16x2(s_acc[2 * kk + 1][0], s_acc[2 * kk + 1][1]);
            p_frag[kk][3] = pack_bf16x2(s_acc[2 * kk + 1][2], s_acc[2 * kk + 1][3]);
        }

#ifdef V3_DEBUG_DUMP
        if (iteration == 0)
        {
            for (int kk = 0; kk < FC::kP; ++kk)
            {
                const int k_lo = kk * MMA_K + frag_col_base(lane);
                float v[2];
                unpack_bf16x2(p_frag[kk][0], v);
                debug_store(kDebugPOffset, row_lo, k_lo, kDebugStride, v[0]);
                debug_store(kDebugPOffset, row_lo, k_lo + 1, kDebugStride, v[1]);
                unpack_bf16x2(p_frag[kk][1], v);
                debug_store(kDebugPOffset, row_hi, k_lo, kDebugStride, v[0]);
                debug_store(kDebugPOffset, row_hi, k_lo + 1, kDebugStride, v[1]);
                unpack_bf16x2(p_frag[kk][2], v);
                debug_store(kDebugPOffset, row_lo, k_lo + 8, kDebugStride, v[0]);
                debug_store(kDebugPOffset, row_lo, k_lo + 9, kDebugStride, v[1]);
                unpack_bf16x2(p_frag[kk][3], v);
                debug_store(kDebugPOffset, row_hi, k_lo + 8, kDebugStride, v[0]);
                debug_store(kDebugPOffset, row_hi, k_lo + 9, kDebugStride, v[1]);
            }
        }
#endif

        // -- PV: O += P * V over head_dim n8 blocks -------------------------
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

        // With a single V stage the next iteration rewrites what PV just read.
        if (kVStages == 1) __syncthreads();
    }

    // =======================================================================
    // Epilogue: normalise and write back in the C layout. Rows past seq_len
    // (a partial query tile) are never stored.
    // =======================================================================
    const int row_lo = warp_row_origin + frag_row_lo(lane);
    const int row_hi = warp_row_origin + frag_row_hi(lane);
    const float inv_lo = running_sum[0] > 0.0f ? 1.0f / running_sum[0] : 0.0f;
    const float inv_hi = running_sum[1] > 0.0f ? 1.0f / running_sum[1] : 0.0f;

    __nv_bfloat16* out_lo = O + layer_offset + static_cast<size_t>(row_lo) * head_dim;
    __nv_bfloat16* out_hi = O + layer_offset + static_cast<size_t>(row_hi) * head_dim;

    for (int n = 0; n < FC::kO; ++n)
    {
        const int col = n * MMA_N + frag_col_base(lane);
        if (row_lo < seq_len)
            *reinterpret_cast<__nv_bfloat162*>(out_lo + col) =
                __floats2bfloat162_rn(o_acc[n][0] * inv_lo, o_acc[n][1] * inv_lo);
        if (row_hi < seq_len)
            *reinterpret_cast<__nv_bfloat162*>(out_hi + col) =
                __floats2bfloat162_rn(o_acc[n][2] * inv_hi, o_acc[n][3] * inv_hi);
    }
}

#ifdef V3_DEBUG_DUMP
int g_v3_debug_dumps = 0;

void write_dump(const char* path, const float* data, int rows, int cols, int stride)
{
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) return;
    for (int row = 0; row < rows; ++row)
        std::fwrite(data + row * stride, sizeof(float), cols, file);
    std::fclose(file);
}

void dump_v3_debug(int head_dim)
{
    if (g_v3_debug_dumps > 0) return;
    ++g_v3_debug_dumps;
    if (cudaDeviceSynchronize() != cudaSuccess) return;

    const int tile_kv = v3_tile_kv(head_dim);
    float host[kDebugFloats];
    if (cudaMemcpyFromSymbol(host, g_v3_debug, sizeof(host)) != cudaSuccess) return;

    write_dump("v3_dump_s.bin", host + kDebugSOffset, q_rows, tile_kv, kDebugStride);
    write_dump("v3_dump_p.bin", host + kDebugPOffset, q_rows, tile_kv, kDebugStride);
    write_dump("v3_dump_q.bin", host + kDebugQOffset, q_rows, head_dim, kDebugFeatureStride);
    write_dump("v3_dump_k.bin", host + kDebugKOffset, tile_kv, head_dim, kDebugFeatureStride);
    std::printf("v3 debug dump: s/p = 64 x %d, q = 64 x %d, k = %d x %d (fp32, row major)\n",
                tile_kv, head_dim, tile_kv, head_dim);
}
#endif

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
            // 32 KB, so no opt-in is required today; keep the call so a later
            // tile or stage bump does not silently fail to launch.
            cudaFuncSetAttribute(attention_kernel_tensorcore_v3<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            attention_kernel_tensorcore_v3<128><<<grid, blocksize, smem, stream>>>(Q, K, V, O, shape.seq_len, scale, causal);
            break;
        }
        default:
            return;
    }

#ifdef V3_DEBUG_DUMP
    dump_v3_debug(shape.head_dim);
#endif
}

FusedTilingInfo query_tensorcore_v3(int head_dim)
{
    FusedTilingInfo info{};
    info.supported = tensorcore_v3_supported(head_dim);
    if (!info.supported) return info;

    info.block_size = blocksize;
    info.tile_kv = v3_tile_kv(head_dim);
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
    // Watch this one at d=128: ~144 registers per lane is close enough to the
    // 255 ceiling that the compiler may spill for scheduling.
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

// Algorithmic DRAM traffic. Same accounting as V2b, but with q_rows=64 and the
// per-head_dim K/V tile the re-read factor drops 8x, which is the whole point of
// the version.
double tensorcore_v3_hbm_bytes(const AttentionShape& shape, bool causal)
{
    const double element = static_cast<double>(sizeof(__nv_bfloat16));
    const int tile_kv = v3_tile_kv(shape.head_dim);
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
        kv_tiles += static_cast<double>((key_limit + tile_kv) / tile_kv);
    }

    bytes += 2.0 * slabs * kv_tiles * tile_kv * shape.head_dim * element;

    return bytes;
}

}  // namespace cuda_op_lab::flash_attention
