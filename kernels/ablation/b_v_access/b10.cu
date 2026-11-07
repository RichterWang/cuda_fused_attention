#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

#include "flash_attention.h"

#define blocksize 256
#define WRAP 32
#define tileKV 64

namespace cuda_op_lab::flash_attention{

namespace{
// kernel private function state here
// dynamic shared mem(128 spill)
extern __shared__ char dynamic_smem[];

constexpr int fusedtiling_smem_bytes_b10(int head_dim)
{
    return (blocksize / WRAP) * head_dim * 4 + 2 * tileKV * head_dim * 2 + tileKV * head_dim * 2 + (blocksize / WRAP) * tileKV * 4;
}
// which : smem64 = 28672 / smem128 = 55296

// cp.async function:
__device__ __forceinline__ void cp_async_16_zfill(void *sharedptr, const void *baseptr, bool valid)
{
#if __CUDA_ARCH__ >= 800
    const unsigned shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(sharedptr));
    const int src_size = valid ? 16 : 0;

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2; \n"
        :
        : "r"(shared_addr), "l"(baseptr), "r"(src_size)
    );
#else
    *reinterpret_cast<uint4*>(sharedptr) = valid ? *reinterpret_cast<const uint4*>(baseptr) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// warp_shuffle reduce function
__forceinline__ __device__  float warp_reduce_max(float value)
{
    for(int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

__forceinline__ __device__ float warp_reduce_sum(float value)
{
    for(int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// online softmax read bf16 tool
template <int N>
__device__ __forceinline__ void load_bf16_vec(float (&out)[N], const __nv_bfloat16* src)
{
    #pragma unroll
    for(int i = 0; i < N; i++) out[i] = __bfloat162float(src[i]);
}

// swizlle function
template<int head_dim>
__forceinline__ __device__ int swz(int row, int chunk)
{
    constexpr int k_chunks = head_dim / 8;
    return (chunk ^ ((row >> 1) & (k_chunks - 1))) * 8;
}

// main kernel with templete
template <int head_dim>
void __global__ attention_kernel_fusedtiling_b10(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O, int seq_len, float scale, bool casual)
{
    // basic data define
    constexpr int krow = blocksize / WRAP;
    constexpr int element_per_lane = head_dim / WRAP;

    // sharedmem allocate
    float* sharedQ = reinterpret_cast<float*>(dynamic_smem);
    __nv_bfloat16* sharedK = reinterpret_cast<__nv_bfloat16*>(sharedQ + krow * head_dim); // double buffering
    __nv_bfloat16* sharedV = (sharedK + 2 * tileKV * head_dim);
    float* score = reinterpret_cast<float*>(sharedV + tileKV * head_dim); // krow * tileKV

    // define the index
    int warp_index = threadIdx.x / WRAP;
    int lane_index = threadIdx.x % WRAP;
    int bh = blockIdx.y;

    // current offset
    size_t current_layer_offset = static_cast<size_t>(bh) * head_dim * seq_len;

    // load Q tile
    #pragma unroll
    for(int i = 0; i < (head_dim + WRAP * 4 - 1) / (WRAP * 4); i++)
    {
        int global_row = blockIdx.x * krow + warp_index;
        int col = lane_index * 4 + i * WRAP * 4;
        int shared_row = warp_index;

        if(global_row < seq_len && col < head_dim)
        {
            uint2 raw = *reinterpret_cast<const uint2*>(&Q[current_layer_offset + global_row * head_dim + col]);
            const float2 vec_1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
            const float2 vec_2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));

            *reinterpret_cast<float4*>(&sharedQ[shared_row * head_dim + col]) = make_float4(
                vec_1.x * scale, vec_1.y * scale, vec_2.x * scale, vec_2.y * scale
            ); // 4 phase, no bank conflict
        }
    }

    // casual mask param
    const int q_max = krow * blockIdx.x + krow - 1;
    const int key_limit = casual ? min(q_max, seq_len - 1) : (seq_len - 1);
    const int nums_loop = (key_limit + tileKV) / tileKV;

    // online softmax param
    float running_max = -FLT_MAX;
    float running_sum = 0.0f; // output demonitor each thread hold one
    float accumulator[element_per_lane] = {0.0f};

    // cp.async write in
    // avoid bank conflict swizzle param
    constexpr int  k_chunks = head_dim / 8;
    const int tid = threadIdx.x;

    // write in of V param
    constexpr int kVecsPerRow = head_dim / 8;
    constexpr int kVecsPerThr = (tileKV * kVecsPerRow) / blocksize;

    // first write in of K and swizzle
    #pragma unroll
    for (int i = 0; i < kVecsPerThr; i++)
    {
        const int v = tid + i * blocksize;
        const int shared_row = v / k_chunks;
        const int chunk = v % k_chunks;
        const bool valid = (shared_row <= key_limit);
        const int safe_row = valid ? shared_row : 0;

        cp_async_16_zfill(&sharedK[shared_row * head_dim + swz<head_dim>(shared_row, chunk)], &K[current_layer_offset + safe_row * head_dim + chunk * 8], valid);
    }
    cp_async_commit();

    // outer loop，each time 64 of sequence
    for(int iteration = 1; iteration < nums_loop; iteration ++)
    {
        int next_group = iteration & 1;
        int current_group = (iteration - 1) & 1;

        // move current V to sharedmem
        #pragma unroll
        for(int i = 0; i < kVecsPerThr; i++)
        {
            const int v = tid + i * blocksize;
            const int shared_row = v / kVecsPerRow;
            const int chunk = v % kVecsPerRow;
            const int col = chunk * 8;
            const int global_row = (iteration - 1) * tileKV + shared_row;
            const bool valid = (global_row <= key_limit);
            const int safe_row = valid ? global_row : 0;

            cp_async_16_zfill(&sharedV[shared_row * head_dim + swz<head_dim>(shared_row, chunk)], &V[current_layer_offset + safe_row * head_dim + col], valid);
        }
        cp_async_commit();

        // move next K to sharedmem
        #pragma unroll
        for(int i = 0; i < kVecsPerThr; i++)
        {
            const int v = tid + i * blocksize;
            const int shared_row = v / k_chunks;
            const int chunk = v % k_chunks;
            const int global_row = iteration * tileKV + shared_row;
            const bool valid = (global_row <= key_limit);
            const int safe_row = valid ? global_row : 0;

            cp_async_16_zfill(&sharedK[next_group * (tileKV * head_dim) + shared_row * head_dim + swz<head_dim>(shared_row, chunk)], &K[current_layer_offset + safe_row * head_dim + chunk * 8], valid);
        }
        cp_async_commit();

        cp_async_wait<2>();
        __syncthreads();

        // caculate the score
        int score_row_thread = warp_index;
        int score_col_thread = lane_index * 2;
        int key = lane_index & (k_chunks - 1); // swizzle key
        float score_sum[2] = {0.0f}; // each thread caculate 2 score elements

        #pragma unroll
        for(int offset_index = 0; offset_index < k_chunks; offset_index ++)
        {
            const int reference_col = (offset_index ^ key) * 8;

            const float4 vec_qa = *reinterpret_cast<const float4*>(&sharedQ[score_row_thread * head_dim + offset_index * 8]);
            const float4 vec_qb = *reinterpret_cast<const float4*>(&sharedQ[score_row_thread * head_dim + offset_index * 8 + 4]);

            #pragma unroll
            for(int j = 0; j < 2; j++)
            {
                uint4 raw = *reinterpret_cast<const uint4*>(&sharedK[current_group * (tileKV * head_dim) + (score_col_thread + j) * head_dim + reference_col]);
                const __nv_bfloat162* ptr = reinterpret_cast<const __nv_bfloat162*>(&raw);

                const float2 k0 = __bfloat1622float2(ptr[0]);
                const float2 k1 = __bfloat1622float2(ptr[1]);
                const float2 k2 = __bfloat1622float2(ptr[2]);
                const float2 k3 = __bfloat1622float2(ptr[3]);

                // reduce and sumup
                score_sum[j] += vec_qa.x * k0.x + vec_qa.y * k0.y + vec_qa.z * k1.x + vec_qa.w * k1.y + vec_qb.x * k2.x + vec_qb.y * k2.y + vec_qb.z * k3.x + vec_qb.w * k3.y;
            }
        }

        *reinterpret_cast<float2*>(&score[score_row_thread * tileKV + score_col_thread]) = make_float2(score_sum[0], score_sum[1]);

        __syncthreads();
        cp_async_wait<1>();
        __syncthreads();

        // online softmax        
        {
            int tiled_row = warp_index; // Q row of the wrap
            int tiled_col = lane_index * 2; // kv col of the wrap

            float2 value = *reinterpret_cast<float2*>((&score[tiled_row * tileKV + tiled_col]));

            const float last_max = running_max;
            float m_per_thread_reduce = fmaxf(value.x, value.y);
            running_max = fmaxf(running_max, warp_reduce_max(m_per_thread_reduce));
            // exp(xi - m_last) * exp(m_last - m_new)
            const float rescale = __expf(last_max - running_max);

            // get current score
            float2 temp_per_thread_score = make_float2(
                __expf(value.x - running_max),
                __expf(value.y - running_max)
            );

            // write back to score to wait for mutipling v
            *reinterpret_cast<float2*>(&score[tiled_row * tileKV + tiled_col]) = temp_per_thread_score;

            //get demonitor
            running_sum = running_sum * rescale + warp_reduce_sum(temp_per_thread_score.x + temp_per_thread_score.y);

            // PV accumulate
            const int d_base = lane_index * element_per_lane;

            // preprocess the accumulator
            #pragma unroll
            for(int i = 0; i < element_per_lane; i++) accumulator[i] *= rescale;

            __syncwarp();

            for(int kr = 0; kr < tileKV; kr++)
            {
                const float p = score[tiled_row * tileKV + kr];

                float temp_v[element_per_lane];
                load_bf16_vec(temp_v, &sharedV[kr * head_dim + swz<head_dim>(kr, d_base / 8) + d_base % 8]);

                #pragma unroll
                for(int i = 0; i < element_per_lane; ++i) accumulator[i] += p * temp_v[i];
            }

            __syncwarp();
        }

        __syncthreads(); // wait for next K and V write in
    }

    // cacualte final round offset
    const int offset = (nums_loop - 1) & 1;

    // move last V to sharedmem
    #pragma unroll
    for(int i = 0; i < kVecsPerThr; i++)
    {
        const int v = tid + i * blocksize;
        const int shared_row = v / kVecsPerRow;
        const int chunk = v % kVecsPerRow;
            const int col = chunk * 8;
        const int global_row = (nums_loop - 1) * tileKV + shared_row;
        const bool valid = (global_row <= key_limit);
        const int safe_row = valid ? global_row : 0; 

        cp_async_16_zfill(&sharedV[shared_row * head_dim + swz<head_dim>(shared_row, chunk)], &V[current_layer_offset + safe_row * head_dim + col], valid);
    }
    cp_async_commit();

    cp_async_wait<1>();
    __syncthreads();

    // caculate the last round score
    int score_row_thread = warp_index;
    int score_col_thread = lane_index * 2;
    int key = lane_index & (k_chunks - 1); // swizzle key
    float score_sum[2] = {0.0f}; // each thread caculate 2 score elements

    #pragma unroll
    for(int offset_index = 0; offset_index < k_chunks; offset_index ++)
    {
        const int reference_col = (offset_index ^ key) * 8;

        const float4 vec_qa = *reinterpret_cast<const float4*>(&sharedQ[score_row_thread * head_dim + offset_index * 8]);
        const float4 vec_qb = *reinterpret_cast<const float4*>(&sharedQ[score_row_thread * head_dim + offset_index * 8 + 4]);

        #pragma unroll
        for(int j = 0; j < 2; j++)
        {
            uint4 raw = *reinterpret_cast<const uint4*>(&sharedK[offset * (tileKV * head_dim) + (score_col_thread + j) * head_dim + reference_col]);
            const __nv_bfloat162* ptr = reinterpret_cast<const __nv_bfloat162*>(&raw);

            const float2 k0 = __bfloat1622float2(ptr[0]);
            const float2 k1 = __bfloat1622float2(ptr[1]);
            const float2 k2 = __bfloat1622float2(ptr[2]);
            const float2 k3 = __bfloat1622float2(ptr[3]);

            // reduce and sumup
            score_sum[j] += vec_qa.x * k0.x + vec_qa.y * k0.y + vec_qa.z * k1.x + vec_qa.w * k1.y + vec_qb.x * k2.x + vec_qb.y * k2.y + vec_qb.z * k3.x + vec_qb.w * k3.y;
        }
    }

    // write into score & mask（only the final tile）
    const int global_qr = blockIdx.x * krow + warp_index;
    const int kv_limit = casual ? min(global_qr, seq_len - 1) : (seq_len - 1);
    const int kr0 = (nums_loop - 1) * tileKV + score_col_thread;
    const int kr1 = kr0 + 1;

    if(kr0 > kv_limit) score_sum[0] = -FLT_MAX;
    if(kr1 > kv_limit) score_sum[1] = -FLT_MAX;

    *reinterpret_cast<float2*>(&score[score_row_thread * tileKV + score_col_thread]) = make_float2(score_sum[0], score_sum[1]);

    __syncthreads();
    cp_async_wait<0>();
    __syncthreads();

    // final round online softmax
    {
        int tiled_row = warp_index; // Q row of the wrap
        int tiled_col = lane_index * 2; // kv col of the wrap

        float2 value = *reinterpret_cast<float2*>((&score[tiled_row * tileKV + tiled_col]));

        const float last_max = running_max;
        float m_per_thread_reduce = fmaxf(value.x, value.y);
        running_max = fmaxf(running_max, warp_reduce_max(m_per_thread_reduce));
        // exp(xi - m_last) * exp(m_last - m_new)
        const float rescale = __expf(last_max - running_max);

        // get current score
        float2 temp_per_thread_score = make_float2(
            __expf(value.x - running_max),
            __expf(value.y - running_max)
        );

        //get demonitor
        running_sum = running_sum * rescale + warp_reduce_sum(temp_per_thread_score.x + temp_per_thread_score.y);

        // write back to shared
        *reinterpret_cast<float2*>(&score[tiled_row * tileKV + tiled_col]) = temp_per_thread_score;

        __syncwarp();

        // PV accumulate
        const int d_base = lane_index * element_per_lane;

        #pragma unroll
        for(int i = 0; i < element_per_lane; i++) accumulator[i] *= rescale;

        for(int kr = 0; kr < tileKV; kr++)
        {
            const float p = score[tiled_row * tileKV + kr];

            float temp_v[element_per_lane];
            load_bf16_vec(temp_v, &sharedV[kr * head_dim + swz<head_dim>(kr, d_base / 8) + d_base % 8]);

            #pragma unroll
            for(int i = 0; i < element_per_lane; ++i) accumulator[i] += p * temp_v[i];
        }

        __syncwarp();
    }

    // write back to output
    const int out_row = krow * blockIdx.x + warp_index;

    const float inv = 1.0f / running_sum;

    if (out_row < seq_len) {
        __nv_bfloat16* destination = O + current_layer_offset + static_cast<size_t>(out_row) * head_dim;
        if constexpr (element_per_lane == 2) {
            *reinterpret_cast<__nv_bfloat162*>(destination + 2 * lane_index) = __floats2bfloat162_rn(accumulator[0] * inv, accumulator[1] * inv);
        } else {
            __nv_bfloat162* d2 = reinterpret_cast<__nv_bfloat162*>(destination + 4 * lane_index);
            d2[0] = __floats2bfloat162_rn(accumulator[0] * inv, accumulator[1] * inv);
            d2[1] = __floats2bfloat162_rn(accumulator[2] * inv, accumulator[3] * inv);
        }
    }

    __syncthreads();
}

}

// public interface function state here
void launch_attention_fusedtiling_b10(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O, const AttentionShape& shape, bool casual, cudaStream_t stream)
{
    constexpr int krow = blocksize / WRAP;

    const dim3 grid(static_cast<size_t>((shape.seq_len + krow - 1) / krow), static_cast<size_t>(shape.batch * shape.heads));

    const float scale = default_scale(shape.head_dim);

    switch(shape.head_dim)
    {
        case 64 : {
            constexpr int smem = fusedtiling_smem_bytes_b10(64);
            attention_kernel_fusedtiling_b10<64><<<grid, blocksize, smem, stream>>>(Q, K, V, O, shape.seq_len, scale, casual);
            break;
        }
        case 128 : {
            constexpr int smem = fusedtiling_smem_bytes_b10(128);
            cudaFuncSetAttribute(attention_kernel_fusedtiling_b10<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); // reset shared mem upper
            attention_kernel_fusedtiling_b10<128><<<grid, blocksize, smem, stream>>>(Q, K, V, O, shape.seq_len, scale, casual);
            break;
        }
        default : return;
    }
}

// static data get of the v1 fused kernel
FusedTilingInfo query_fused_tiling_b10(int head_dim)
{
    FusedTilingInfo info{};
    info.supported = (head_dim == 64 || head_dim == 128);
    if (!info.supported) return info;

    info.block_size = blocksize;
    info.tile_kv = tileKV;
    info.rows_per_block = blocksize / WRAP;

    const int dynamic_bytes = (head_dim == 64) ? fusedtiling_smem_bytes_b10(64) : fusedtiling_smem_bytes_b10(128);
    const void* func = (head_dim == 64) ? reinterpret_cast<const void*>(attention_kernel_fusedtiling_b10<64>) : reinterpret_cast<const void*>(attention_kernel_fusedtiling_b10<128>);

    if (dynamic_bytes > 48 * 1024) cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);

    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(&attributes, func);

    info.shared_bytes_per_block = static_cast<int>(attributes.sharedSizeBytes) + dynamic_bytes;
    info.registers_per_thread = attributes.numRegs;
    info.local_bytes_per_thread = static_cast<int>(attributes.localSizeBytes);

    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, func, blocksize, dynamic_bytes);
    info.max_blocks_per_sm = blocks;

    // Report the budget blocks/SM is competing for
    int device = 0;
    cudaGetDevice(&device);
    int shared_per_sm = 0;
    cudaDeviceGetAttribute(&shared_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    info.device_shared_bytes_per_sm = shared_per_sm;
    info.shared_limited_blocks_per_sm = (info.shared_bytes_per_block > 0) ? shared_per_sm / info.shared_bytes_per_block : 0;

    return info;
}

// v2 traffic: 
double fused_tiling_hbm_bytes_b10(const AttentionShape& shape, bool causal)
{
    constexpr int krow = blocksize / WRAP;

    const double element = static_cast<double>(sizeof(__nv_bfloat16));
    const int slabs = shape.batch * shape.heads;
    const int row_blocks = (shape.seq_len + krow - 1) / krow;

    // Q read once, O written once.
    double bytes = 2.0 * slabs * shape.seq_len * shape.head_dim * element;

    // K/V tiles, summed over the blocks that actually load them.
    double kv_tiles = 0.0;
    for (int block = 0; block < row_blocks; ++block)
    {
        const int q_max = block * krow + krow - 1;
        const int key_limit = causal ? ((q_max < shape.seq_len - 1) ? q_max : shape.seq_len - 1) : (shape.seq_len - 1);
        kv_tiles += static_cast<double>((key_limit + tileKV) / tileKV);
    }

    bytes += 2.0 * slabs * kv_tiles * tileKV * shape.head_dim * element;

    return bytes;
}

}
