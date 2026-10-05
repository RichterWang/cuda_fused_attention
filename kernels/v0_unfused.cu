// v0: unfused attention. Three kernels, S materialized in global memory.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

#include "flash_attention.h"

namespace cuda_op_lab::flash_attention {

namespace {

// define score matrix block dim    
constexpr int kScoreBlockDim = 16;   // 16x16 = 256
// define softmax block dim
constexpr int kSoftmaxBlockSize = 256;

// S[bh] = Q[bh] @ K[bh]^T * scale. where masked set to -inf
__global__ void scores_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, float* __restrict__ s, int seq_len, int head_dim, float scale, bool causal)
{
    const int query = blockIdx.y * kScoreBlockDim + threadIdx.y;
    const int key = blockIdx.x * kScoreBlockDim + threadIdx.x;
    const int bh = blockIdx.z; // which batch-head

    if (query >= seq_len || key >= seq_len) return;

    // when offset or ptr pos caculate, use size_t
    // offset of QKVO
    const size_t slab = static_cast<size_t>(bh) * seq_len * head_dim;
    // offset of S
    const size_t score_slab = static_cast<size_t>(bh) * seq_len * seq_len;

    // casual mask, done on score lab
    if (causal && key > query)
    {
        s[score_slab + static_cast<size_t>(query) * seq_len + key] = -FLT_MAX;
        return;
    }

    // get index of caculate
    const __nv_bfloat16* q_row = q + slab + static_cast<size_t>(query) * head_dim;
    const __nv_bfloat16* k_row = k + slab + static_cast<size_t>(key) * head_dim;

    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) dot += __bfloat162float(q_row[dim]) * __bfloat162float(k_row[dim]);

    s[score_slab + static_cast<size_t>(query) * seq_len + key] = dot * scale;
}

// Row-wise softmax in place, one block per row, shared-memory tree reduction.
__global__ void softmax_rows_kernel(float* __restrict__ s, int cols)
{
    extern __shared__ float scratch[]; // dynamic allocation of sharedmem size, defined by num of hhreads in a row

    float* row = s + static_cast<size_t>(blockIdx.x) * cols;
    const int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) local_max = fmaxf(local_max, row[col]);

    scratch[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    const float row_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x)
    {
        const float value = __expf(row[col] - row_max);
        row[col] = value;
        local_sum += value;
    }

    scratch[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }

    const float denominator = scratch[0] > 0.0f ? scratch[0] : 1.0f;
    const float inverse = 1.0f / denominator;
    __syncthreads();

    for (int col = tid; col < cols; col += blockDim.x) row[col] *= inverse;
}

// O[bh] = P[bh] @ V[bh]. One thread per output element, summing over seq_len.
__global__ void pv_kernel(const float* __restrict__ p, const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o, int seq_len, int head_dim)
{
    const int query = blockIdx.y * kScoreBlockDim + threadIdx.y;
    const int dim = blockIdx.x * kScoreBlockDim + threadIdx.x;
    const int bh = blockIdx.z;

    if (query >= seq_len || dim >= head_dim) return;

    const size_t slab = static_cast<size_t>(bh) * seq_len * head_dim;
    const size_t score_slab = static_cast<size_t>(bh) * seq_len * seq_len;

    const float* p_row = p + score_slab + static_cast<size_t>(query) * seq_len;

    float accumulator = 0.0f;
    for (int key = 0; key < seq_len; ++key) accumulator += p_row[key] * __bfloat162float(v[slab + static_cast<size_t>(key) * head_dim + dim]);

    o[slab + static_cast<size_t>(query) * head_dim + dim] = __float2bfloat16(accumulator);
}

}  // namespace

double attention_flops(const AttentionShape& shape, bool causal)
{
    const double n = static_cast<double>(shape.seq_len);
    // the k q pair that need to caculate
    const double pairs = causal ? n * (n + 1.0) / 2.0 : n * n;

    // Two GEMMs, each 2 FLOP (multiply + add) per (query, key, dim) triple.
    return 4.0 * pairs * static_cast<double>(shape.head_dim) * static_cast<double>(shape.batch) * static_cast<double>(shape.heads);
}

size_t unfused_workspace_bytes(const AttentionShape& shape)
{
    return static_cast<size_t>(shape.batch) * shape.heads * shape.seq_len * shape.seq_len * sizeof(float);
}

void launch_softmax_rows_inplace(float* s, int rows, int cols, cudaStream_t stream)
{
    const dim3 grid(static_cast<unsigned>(rows));
    const dim3 block(kSoftmaxBlockSize);
    softmax_rows_kernel<<<grid, block, kSoftmaxBlockSize * sizeof(float), stream>>>(s, cols);
}

void launch_attention_unfused(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float* workspace, const AttentionShape& shape, bool causal, cudaStream_t stream)
{
    const int bh_count = shape.batch * shape.heads;
    const float scale = default_scale(shape.head_dim);

    const dim3 score_block(kScoreBlockDim, kScoreBlockDim);
    const dim3 score_grid((shape.seq_len + kScoreBlockDim - 1) / kScoreBlockDim, (shape.seq_len + kScoreBlockDim - 1) / kScoreBlockDim, static_cast<unsigned>(bh_count));
    scores_kernel<<<score_grid, score_block, 0, stream>>>(q, k, workspace, shape.seq_len, shape.head_dim, scale, causal);

    // The softmax treats the whole workspace as [bh*N, N] matrix. Rows never
    launch_softmax_rows_inplace(workspace, bh_count * shape.seq_len, shape.seq_len, stream);

    const dim3 pv_block(kScoreBlockDim, kScoreBlockDim);
    const dim3 pv_grid((shape.head_dim + kScoreBlockDim - 1) / kScoreBlockDim, (shape.seq_len + kScoreBlockDim - 1) / kScoreBlockDim, static_cast<unsigned>(bh_count));
    pv_kernel<<<pv_grid, pv_block, 0, stream>>>(workspace, v, o, shape.seq_len, shape.head_dim);
}

}  // namespace cuda_op_lab::flash_attention
