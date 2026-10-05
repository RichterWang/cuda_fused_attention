// v1: fused attention, one block per query row, online softmax.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

#include "flash_attention.h"

namespace cuda_op_lab::flash_attention {

namespace {

constexpr int kBlockSize = 128; // block size(threads)

// get sharedmem size data
template <int HeadDim>
constexpr int tile_rows_for()
{
    return 4096 / HeadDim;
}

// padding to aviod bank conflict
constexpr int kPad = 1;

// warp shuffle reduce getmax & getsum
__forceinline__ __device__ float warp_reduce_max(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

__forceinline__ __device__ float warp_reduce_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// main kernel
template <int HeadDim>
__global__ void fused_row_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o, int seq_len, float scale, bool causal)
{
    constexpr int kWarps = kBlockSize / 32;
    constexpr int kDimsPerLane = HeadDim / 32; // how many data process per thread
    constexpr int kTileN = tile_rows_for<HeadDim>(); // shared_mem of tiled K and V row num

    // K and V tiles, plus scratch for the cross-warp reductions. 
    __shared__ float k_tile[kTileN][HeadDim + kPad];
    __shared__ float v_tile[kTileN][HeadDim];
    __shared__ float probabilities[kTileN]; // scores
    __shared__ float scratch[kWarps]; // reduce buffer
    __shared__ float q_shared[HeadDim];
    __shared__ float output[HeadDim]; // output buffer

    // get the offset index
    const int query = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    if (query >= seq_len) return;

    // get current offset of the data of this layer of matrix
    const size_t slab = static_cast<size_t>(bh) * seq_len * HeadDim;

    // boardcast
    for (int index = tid; index < HeadDim; index += kBlockSize) q_shared[index] = __bfloat162float(q[slab + static_cast<size_t>(query) * HeadDim + index]);
    __syncthreads();

    // online softmax + accmulator
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float accumulator[kDimsPerLane]; // num of process data of a lane(thread)
    for (int index = 0; index < kDimsPerLane; ++index) accumulator[index] = 0.0f; // init of per lane

    // early stop by casual mask
    const int key_limit = causal ? query + 1 : seq_len;

    for (int tile_start = 0; tile_start < key_limit; tile_start += kTileN)
    {
        const int tile_rows = min(kTileN, key_limit - tile_start);

        // tile load
        for (int index = tid; index < tile_rows * HeadDim; index += kBlockSize)
        {
            const int row = index / HeadDim;
            const int dim = index % HeadDim;
            k_tile[row][dim] = __bfloat162float(k[slab + static_cast<size_t>(tile_start + row) * HeadDim + dim]);
            v_tile[row][dim] = __bfloat162float(v[slab + static_cast<size_t>(tile_start + row) * HeadDim + dim]);
        }
        __syncthreads();

        // each thread hold a line of the tile(a dot in the Score matrix)
        // each block hold a row of S
        float score = -FLT_MAX;
        const int score_row = tid;
        if (score_row < tile_rows)
        {
            float dot = 0.0f;
            for (int dim = 0; dim < HeadDim; ++dim) dot += q_shared[dim] * k_tile[score_row][dim];
            score = dot * scale;
        }

        // reduce of max, for the hole block
        float tile_max = warp_reduce_max(score);
        if (lane == 0) scratch[warp] = tile_max;
        __syncthreads();
        tile_max = scratch[0]; // simple init
        for (int index = 1; index < kWarps; ++index) tile_max = fmaxf(tile_max, scratch[index]);

        const float new_max = fmaxf(running_max, tile_max); // current max update
        const float rescale = __expf(running_max - new_max); // current rescale

        // only update current and before round
        const float probability = score_row < tile_rows ? __expf(score - new_max) : 0.0f;

        // reduce sum
        float tile_sum = warp_reduce_sum(probability);
        __syncthreads();  // scratch is reused, so the previous read must be done
        if (lane == 0) scratch[warp] = tile_sum;
        __syncthreads();
        tile_sum = 0.0f;
        for (int index = 0; index < kWarps; ++index) tile_sum += scratch[index];

        running_sum = running_sum * rescale + tile_sum;
        running_max = new_max;

        // write the probility into the sharedmem
        if (score_row < tile_rows) probabilities[score_row] = probability;
        __syncthreads();

        for (int index = 0; index < kDimsPerLane; ++index)
        {
            const int dim = index * 32 + lane;
            float partial = 0.0f;

            for (int row = warp; row < tile_rows; row += kWarps) partial += probabilities[row] * v_tile[row][dim];
            accumulator[index] = accumulator[index] * rescale + partial;
        }
        __syncthreads();
    }

    // make the softmax divide at last
    if (warp == 0) for (int index = 0; index < kDimsPerLane; ++index) output[index * 32 + lane] = 0.0f;
    __syncthreads();

    for (int source = 0; source < kWarps; ++source)
    {
        if (warp == source) for (int index = 0; index < kDimsPerLane; ++index) output[index * 32 + lane] += accumulator[index];
        __syncthreads();
    }

    // divide and write back
    const float inverse = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    for (int index = tid; index < HeadDim; index += kBlockSize) o[slab + static_cast<size_t>(query) * HeadDim + index] = __float2bfloat16(output[index] * inverse);
}

}  // namespace

// check if the input data fits the kernel
bool fused_row_supported(int head_dim)
{
    return head_dim == 64 || head_dim == 128;
}

void launch_attention_fused_row(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const AttentionShape& shape, bool causal, cudaStream_t stream)
{
    if (!fused_row_supported(shape.head_dim)) return;

    const dim3 grid(static_cast<unsigned>(shape.seq_len), static_cast<unsigned>(shape.batch * shape.heads));
    const float scale = default_scale(shape.head_dim);

    if (shape.head_dim == 64) fused_row_kernel<64><<<grid, kBlockSize, 0, stream>>>(q, k, v, o, shape.seq_len, scale, causal);
    else fused_row_kernel<128><<<grid, kBlockSize, 0, stream>>>(q, k, v, o, shape.seq_len, scale, causal);
}

// static data get of the v1 fused kernel
FusedRowInfo query_fused_row(int head_dim)
{
    FusedRowInfo info{};
    info.supported = fused_row_supported(head_dim);
    if (!info.supported) return info;

    info.block_size = kBlockSize;
    info.tile_n = head_dim == 64 ? tile_rows_for<64>() : tile_rows_for<128>();

    cudaFuncAttributes attributes{};
    if (head_dim == 64) cudaFuncGetAttributes(&attributes, reinterpret_cast<const void*>(fused_row_kernel<64>));
    else cudaFuncGetAttributes(&attributes, reinterpret_cast<const void*>(fused_row_kernel<128>));

    info.shared_bytes_per_block = static_cast<int>(attributes.sharedSizeBytes);
    info.registers_per_thread = attributes.numRegs;
    info.local_bytes_per_thread = static_cast<int>(attributes.localSizeBytes);

    int blocks = 0;
    if (head_dim == 64) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reinterpret_cast<const void*>(fused_row_kernel<64>), kBlockSize, 0);
    else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reinterpret_cast<const void*>(fused_row_kernel<128>), kBlockSize, 0);
    info.max_blocks_per_sm = blocks;

    return info;
}

}  // namespace cuda_op_lab::flash_attention
