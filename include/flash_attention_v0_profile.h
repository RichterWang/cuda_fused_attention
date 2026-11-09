#pragma once

#include "flash_attention.h"

namespace cuda_op_lab::flash_attention {

// Caller owns timing-enabled events and waits for pv_end before reading times.
// All events and launches use the same stream. Workspace matches original V0.
struct V0ProfileEvents
{
    cudaEvent_t qk_start;
    cudaEvent_t qk_end;
    cudaEvent_t softmax_end;
    cudaEvent_t pv_end;
};

// Same valid-shape and pointer preconditions as launch_attention_unfused.
// Returns event-record/launch errors; asynchronous errors surface on stream sync.
// A null events pointer launches exactly three kernels without timing markers.
cudaError_t launch_attention_unfused_profile(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* o, float* workspace, const AttentionShape& shape, bool causal,
    const V0ProfileEvents* events = nullptr, cudaStream_t stream = nullptr);

}  // namespace cuda_op_lab::flash_attention
