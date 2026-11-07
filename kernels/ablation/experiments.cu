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
using cuda_op_lab::flash_attention::launch_attention_fusedtiling_v2b;
using cuda_op_lab::flash_attention::query_fused_tiling_v2b;
using cuda_op_lab::flash_attention::fused_tiling_v2b_hbm_bytes;

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

void run_fused_tiling_v2b(void* context)
{
    Buffers& buffers = *static_cast<Buffers*>(context);
    launch_attention_fusedtiling_v2b(buffers.q, buffers.k, buffers.v, buffers.o, buffers.shape, buffers.causal, nullptr);
}

}  // namespace

#include <fstream>
#include <string>
#include <filesystem>
#include <array>
#include <numeric>
namespace cuda_op_lab::flash_attention {
#define DECL(n) void launch_attention_fusedtiling_##n(const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*,__nv_bfloat16*,const AttentionShape&,bool,cudaStream_t)
DECL(b01); DECL(b10); DECL(c1_sync); DECL(c1_async); DECL(v2b_single);
#undef DECL
}
using namespace cuda_op_lab::flash_attention;
using Launcher=void(*)(const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*,__nv_bfloat16*,const AttentionShape&,bool,cudaStream_t);
struct Variant { const char* name; Launcher call; bool v0; };
void v0(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* o,const AttentionShape& s,bool c,cudaStream_t st) { (void)q;(void)k;(void)v;(void)o;(void)s;(void)c;(void)st; }
const Variant variants[]={{"v0",v0,true},{"v2a",launch_attention_fusedtiling,false},{"v2b",launch_attention_fusedtiling_v2b,false},{"b01",launch_attention_fusedtiling_b01,false},{"b10",launch_attention_fusedtiling_b10,false},{"c1_sync",launch_attention_fusedtiling_c1_sync,false},{"c1_async",launch_attention_fusedtiling_c1_async,false},{"v2b_single",launch_attention_fusedtiling_v2b_single,false}};
void launch(const Variant& x,Buffers& b) { if(x.v0) run_unfused(&b); else x.call(b.q,b.k,b.v,b.o,b.shape,b.causal,nullptr); CUDA_CHECK(cudaGetLastError()); }
int main(int argc,char** argv) {
 if(argc==8 && std::string(argv[1])=="--profile") {
  int id=-1; for(int i=0;i<8;i++) if(std::string(argv[2])==variants[i].name) id=i;
  if(id<0 || id==0) return 2;
  AttentionShape sh{std::atoi(argv[3]),std::atoi(argv[4]),std::atoi(argv[5]),std::atoi(argv[6])}; bool causal=std::atoi(argv[7]);
  size_t count=(size_t)sh.batch*sh.heads*sh.seq_len*sh.head_dim,bytes=count*sizeof(__nv_bfloat16);std::vector<__nv_bfloat16> x(count);__nv_bfloat16 *q,*k,*v,*o;
  CUDA_CHECK(cudaMalloc(&q,bytes)); CUDA_CHECK(cudaMalloc(&k,bytes)); CUDA_CHECK(cudaMalloc(&v,bytes)); CUDA_CHECK(cudaMalloc(&o,bytes));
  for(auto p:std::array<std::pair<__nv_bfloat16*,int>,3>{{{q,11},{k,22},{v,33}}}) {fill_normal(x,p.second);CUDA_CHECK(cudaMemcpy(p.first,x.data(),bytes,cudaMemcpyHostToDevice));}
  Buffers buf{q,k,v,o,nullptr,sh,causal}; for(int i=0;i<6;i++)launch(variants[id],buf);CUDA_CHECK(cudaDeviceSynchronize());cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);return 0;
 }
 const int samples=argc>1?std::atoi(argv[1]):30; const int repeats=argc>2?std::atoi(argv[2]):1;
 if(samples<1||repeats<1) return 2;
 std::filesystem::create_directories("kernels/ablation/result");
 std::ofstream correct("kernels/ablation/result/correctness.csv"); correct<<"variant,d,n,causal,input,max_abs,max_row_ulp,nonfinite,status\n";
 bool pass[8]; std::fill(std::begin(pass),std::end(pass),true);
 for(int d:{64,128}) for(int n:{1,7,8,9,31,32,33,63,64,65,127,128,129,256}) for(bool causal:{false,true}) for(int input=0;input<5;input++) {
  AttentionShape s{1,2,n,d}; size_t count=(size_t)2*n*d, bytes=count*sizeof(__nv_bfloat16); std::vector<__nv_bfloat16> q(count),k(count),v(count),o(count); std::vector<double> ref(count);
  fill_normal(q,11);fill_normal(k,22);fill_normal(v,33);
  if(input==1) for(size_t i=0;i<count;i++)q[i]=k[i]=v[i]=__float2bfloat16(0);
  if(input==2) for(size_t i=0;i<count;i++)q[i]=k[i]=v[i]=__float2bfloat16(1);
  if(input==3) for(size_t i=0;i<count;i++)v[i]=__float2bfloat16((int)(i%d)-d/2);
  if(input==4) for(size_t i=0;i<count;i++){q[i]=__float2bfloat16(__bfloat162float(q[i])*4);k[i]=__float2bfloat16(__bfloat162float(k[i])*4);}
  reference_attention(q,k,v,ref,s,causal);
  __nv_bfloat16 *dq,*dk,*dv,*out; float* ws; CUDA_CHECK(cudaMalloc(&dq,bytes));CUDA_CHECK(cudaMalloc(&dk,bytes));CUDA_CHECK(cudaMalloc(&dv,bytes));CUDA_CHECK(cudaMalloc(&out,bytes));CUDA_CHECK(cudaMalloc(&ws,unfused_workspace_bytes(s)));
  CUDA_CHECK(cudaMemcpy(dq,q.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dk,k.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dv,v.data(),bytes,cudaMemcpyHostToDevice)); Buffers buf{dq,dk,dv,out,ws,s,causal};
  for(int z=0;z<8;z++){CUDA_CHECK(cudaMemset(out,0,bytes));launch(variants[z],buf);CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(o.data(),out,bytes,cudaMemcpyDeviceToHost));double maxabs=0,maxulp=0;int bad=0;for(int row=0;row<2*n;row++){double scale=0;for(int j=0;j<d;j++)scale=std::max(scale,std::abs(ref[(size_t)row*d+j]));float ulp=bf16_ulp((float)scale);for(int j=0;j<d;j++){size_t i=(size_t)row*d+j;float got=__bfloat162float(o[i]);if(!std::isfinite(got)){bad++;continue;}maxabs=std::max(maxabs,std::abs(got-ref[i]));float rounded=__bfloat162float(__float2bfloat16((float)ref[i]));maxulp=std::max(maxulp,(double)(std::abs(got-rounded)/ulp));}}bool ok=!bad&&maxulp<=2.0; pass[z]&=ok;correct<<variants[z].name<<','<<d<<','<<n<<','<<causal<<','<<input<<','<<maxabs<<','<<maxulp<<','<<bad<<','<<(ok?"PASS":"FAIL")<<'\n';}
  cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(out);cudaFree(ws);
 }
 correct.close();for(int z=0;z<8;z++)std::printf("%s correctness %s\n",variants[z].name,pass[z]?"PASS":"FAIL");
 struct Case2 {int b,h,n,d;bool c;}; std::vector<Case2> cases={Case2{4,8,512,64,false},Case2{2,8,2048,64,false},Case2{1,8,2048,128,true}};
 for(int group=0;group<4;group++) {const char* directories[]={"a_overall","b_v_access","c_v2a_pipeline","d_v2b_pipeline"}; std::filesystem::create_directories(std::string("kernels/ablation/result/")+directories[group]); std::string file=std::string("kernels/ablation/result/")+directories[group]+"/raw.csv";std::ofstream csv(file);csv<<"sample_id,order,variant,batch,heads,seq_len,head_dim,causal,seed,calls_per_sample,latency_ms,correctness_status,workspace_bytes\n";std::vector<int> ids=group==0?std::vector<int>{0,1,2}:group==1?std::vector<int>{1,3,4,2}:group==2?std::vector<int>{5,6,1}:std::vector<int>{2,7};
 auto group_cases = cases; if(group==3) { group_cases={{4,8,512,64,false},{4,8,512,64,true},{2,8,2048,64,false},{2,8,2048,64,true},{4,8,512,128,false},{4,8,512,128,true},{1,8,2048,128,false},{1,8,2048,128,true}}; } if(group==0) { group_cases.push_back({4,8,512,64,true}); group_cases.push_back({2,8,2048,64,true}); group_cases.push_back({1,8,4096,64,true}); }
 for(auto c:group_cases){AttentionShape s{c.b,c.h,c.n,c.d};size_t count=(size_t)c.b*c.h*c.n*c.d,bytes=count*sizeof(__nv_bfloat16);std::vector<__nv_bfloat16> x(count);__nv_bfloat16 *q,*k,*v,*o;float* ws;CUDA_CHECK(cudaMalloc(&q,bytes));CUDA_CHECK(cudaMalloc(&k,bytes));CUDA_CHECK(cudaMalloc(&v,bytes));CUDA_CHECK(cudaMalloc(&o,bytes));CUDA_CHECK(cudaMalloc(&ws,unfused_workspace_bytes(s)));for(auto p:std::array<std::pair<__nv_bfloat16*,int>,3>{{{q,11},{k,22},{v,33}}}){fill_normal(x,p.second);CUDA_CHECK(cudaMemcpy(p.first,x.data(),bytes,cudaMemcpyHostToDevice));}Buffers buf{q,k,v,o,ws,s,c.c};for(int id:ids)if(pass[id])for(int i=0;i<10;i++)launch(variants[id],buf);CUDA_CHECK(cudaDeviceSynchronize());cudaEvent_t start,stop;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&stop));for(int sample=0;sample<samples;sample++)for(int idx=0;idx<(int)ids.size();idx++){int id=ids[(idx+sample)%ids.size()];if(!pass[id])continue;CUDA_CHECK(cudaEventRecord(start));for(int j=0;j<repeats;j++)launch(variants[id],buf);CUDA_CHECK(cudaEventRecord(stop));CUDA_CHECK(cudaEventSynchronize(stop));float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));csv<<sample<<','<<idx<<','<<variants[id].name<<','<<c.b<<','<<c.h<<','<<c.n<<','<<c.d<<','<<c.c<<",11/22/33,"<<repeats<<','<<ms/repeats<<",PASS,"<<(variants[id].v0?unfused_workspace_bytes(s):0)<<'\n';}cudaEventDestroy(start);cudaEventDestroy(stop);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);cudaFree(ws);std::printf("group %c N=%d D=%d done\n",'A'+group,c.n,c.d);fflush(stdout);}}
 return std::all_of(std::begin(pass),std::end(pass),[](bool x){return x;})?0:1;
}
