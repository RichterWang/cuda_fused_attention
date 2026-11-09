// Unified V0 correctness, stage timing and focused NCU entry point.
#include "flash_attention_v0_profile.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cuda_op_lab::flash_attention;

namespace {
void check(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&ptr, bytes)); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};

int number(const char* text, int minimum)
{
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < minimum || value > INT_MAX)
        throw std::runtime_error(std::string("invalid integer: ") + text);
    return static_cast<int>(value);
}

size_t multiply(size_t a, size_t b)
{
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error("shape size overflows size_t");
    return a * b;
}

struct Data {
    AttentionShape shape;
    size_t count, bytes;
    Buffer<__nv_bfloat16> q, k, v, o, reference;
    Buffer<float> workspace;

    explicit Data(AttentionShape s)
        : shape(s), count(multiply(multiply(multiply(s.batch, s.heads), s.seq_len), s.head_dim)),
          bytes(multiply(count, sizeof(__nv_bfloat16))),
          q(bytes), k(bytes), v(bytes), o(bytes), reference(bytes),
          workspace(multiply(multiply(multiply(multiply(s.batch, s.heads), s.seq_len), s.seq_len), sizeof(float)))
    {
        std::vector<__nv_bfloat16> host(count);
        unsigned seed = 11;
        for (auto* ptr : {q.ptr, k.ptr, v.ptr}) {
            std::mt19937 engine(seed);
            std::normal_distribution<float> distribution(0.0f, 1.0f);
            for (auto& value : host) value = __float2bfloat16(distribution(engine));
            check(cudaMemcpy(ptr, host.data(), bytes, cudaMemcpyHostToDevice));
            seed += 11;
        }
    }

    void original(bool causal, cudaStream_t stream)
    {
        launch_attention_unfused(q.ptr, k.ptr, v.ptr, reference.ptr, workspace.ptr, shape, causal, stream);
        check(cudaGetLastError());
    }

    void profile(bool causal, cudaStream_t stream, const V0ProfileEvents* events = nullptr)
    {
        check(launch_attention_unfused_profile(q.ptr, k.ptr, v.ptr, o.ptr, workspace.ptr,
                                              shape, causal, events, stream));
    }

    void verify(bool causal, cudaStream_t stream)
    {
        original(causal, stream);
        // Verify the instrumented path, including records on a non-default stream.
        Event start, qk, softmax, pv;
        V0ProfileEvents events{start.value, qk.value, softmax.value, pv.value};
        profile(causal, stream, &events);
        check(cudaStreamSynchronize(stream));
        std::vector<__nv_bfloat16> actual(count), expected(count);
        check(cudaMemcpy(actual.data(), o.ptr, bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(expected.data(), reference.ptr, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(__bfloat162float(actual[i])) ||
                !std::isfinite(__bfloat162float(expected[i])) ||
                std::memcmp(&actual[i], &expected[i], sizeof(actual[i])) != 0)
                throw std::runtime_error("profile output differs from original V0 at element " + std::to_string(i));
        }
        // Also exercise the event-free path used by NCU.
        profile(causal, stream);
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(), o.ptr, bytes, cudaMemcpyDeviceToHost));
        if (std::memcmp(actual.data(), expected.data(), bytes) != 0)
            throw std::runtime_error("event-free profile output differs from original V0");
        std::printf("correctness: PASS (finite, bitwise equal to original V0)\n");
    }
};

float elapsed(cudaEvent_t start, cudaEvent_t end)
{
    float ms;
    check(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

float quantile(std::vector<float> values, double p)
{
    std::sort(values.begin(), values.end());
    const double index = (values.size() - 1) * p;
    const size_t lo = static_cast<size_t>(index);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (index - lo);
}

void run(AttentionShape shape, bool causal, int samples, int warmup, bool ncu, bool verify_only)
{
    // V0 uses grid.z for batch*heads and signed int for the softmax row count.
    const size_t bh = multiply(shape.batch, shape.heads);
    if (bh > 65535 || multiply(bh, shape.seq_len) > INT_MAX ||
        shape.seq_len > 65535 * 16 || shape.head_dim > INT_MAX - 15)
        throw std::runtime_error("shape exceeds V0 launch/index limits");
    Data data(shape);
    Stream stream;
    std::printf("shape: B=%d H=%d N=%d D=%d causal=%d workspace_bytes=%zu\n",
                shape.batch, shape.heads, shape.seq_len, shape.head_dim, causal,
                unfused_workspace_bytes(shape));
    if (ncu) {
        // No validation, warmup, or timing events: exactly 3 * samples launches.
        for (int i = 0; i < samples; ++i) data.profile(causal, stream.value);
        check(cudaStreamSynchronize(stream.value));
        std::printf("NCU: %d complete pipelines (QK -> softmax -> PV)\n", samples);
        return;
    }
    data.verify(causal, stream.value);
    if (verify_only) return;
    for (int i = 0; i < warmup; ++i) {
        data.original(causal, stream.value);
        data.profile(causal, stream.value);
    }
    check(cudaStreamSynchronize(stream.value));
    Event start, qk, softmax, pv, begin, end;
    V0ProfileEvents events{start.value, qk.value, softmax.value, pv.value};
    std::array<std::vector<float>, 5> times;
    for (int i = 0; i < samples; ++i) {
        // Independent original-V0 E2E interval: no markers between its kernels.
        check(cudaEventRecord(begin.value, stream.value));
        data.original(causal, stream.value);
        check(cudaEventRecord(end.value, stream.value));
        data.profile(causal, stream.value, &events);
        check(cudaEventSynchronize(pv.value));
        times[0].push_back(elapsed(start.value, qk.value));
        times[1].push_back(elapsed(qk.value, softmax.value));
        times[2].push_back(elapsed(softmax.value, pv.value));
        times[3].push_back(elapsed(start.value, pv.value));
        times[4].push_back(elapsed(begin.value, end.value));
    }
    const char* labels[] = {"QK", "softmax", "PV", "instrumented_pipeline", "original_v0_e2e"};
    std::printf("samples=%d warmup=%d cache=warm/uncontrolled; units=ms\n", samples, warmup);
    std::printf("metric,median_ms,q1_ms,q3_ms,iqr_ms\n");
    for (size_t i = 0; i < times.size(); ++i) {
        const float q1 = quantile(times[i], 0.25), q3 = quantile(times[i], 0.75);
        std::printf("%s,%.6f,%.6f,%.6f,%.6f\n", labels[i], quantile(times[i], 0.5), q1, q3, q3 - q1);
    }
}

void usage(const char* program)
{
    std::printf("Usage:\n  %s                         # two representative dense shapes, 30 samples, 5 warmups\n"
                "  %s B H N D causal [samples [warmup]]\n"
                "  %s --check                 # small/odd/256 lengths, D=64/128, dense/causal\n"
                "  %s --ncu B H N D causal [iterations] # no validation/warmup/events\n",
                program, program, program, program);
}
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::strcmp(argv[1], "--help") == 0) { usage(argv[0]); return 0; }
        const bool ncu = argc > 1 && std::strcmp(argv[1], "--ncu") == 0;
        const bool verify_only = argc == 2 && std::strcmp(argv[1], "--check") == 0;
        const int offset = ncu ? 2 : 1;
        AttentionShape shape{};
        bool causal = false;
        int samples = ncu ? 1 : 30, warmup = 5;
        if (argc != 1 && !verify_only) {
            const int args = argc - offset;
            if (args < 5 || args > (ncu ? 6 : 7)) { usage(argv[0]); return 1; }
            shape = {number(argv[offset], 1), number(argv[offset + 1], 1),
                     number(argv[offset + 2], 1), number(argv[offset + 3], 1)};
            const int mask = number(argv[offset + 4], 0);
            if (mask > 1) throw std::runtime_error("causal must be 0 or 1");
            causal = mask != 0;
            if (args >= 6) samples = number(argv[offset + 5], 1);
            if (args == 7) warmup = number(argv[offset + 6], 0);
        }
        int device;
        check(cudaGetDevice(&device));
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, device));
        std::printf("device: %s (sm_%d%d)\n", props.name, props.major, props.minor);
        if (verify_only) {
            for (int n : {1, 17, 256})
                for (int d : {64, 128})
                    for (bool mask : {false, true}) run({1, 2, n, d}, mask, 1, 0, false, true);
        } else if (argc == 1) {
            run({2, 8, 2048, 64}, false, samples, warmup, false, false);
            run({1, 8, 2048, 128}, false, samples, warmup, false, false);
        } else {
            run(shape, causal, samples, warmup, ncu, false);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
