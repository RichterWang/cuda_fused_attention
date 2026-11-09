cmake_minimum_required(VERSION 3.22)

project(cuda_fused_attention LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
enable_testing()

add_library(flash_attention INTERFACE)
target_include_directories(flash_attention INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(flash_attention INTERFACE CUDA::cudart)

function(add_attention_kernel target_name source_file)
  add_library(${target_name} STATIC ${source_file})
  target_link_libraries(${target_name} PUBLIC flash_attention)
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

add_attention_kernel(flash_attention_v0 kernels/v0_unfused.cu)
add_attention_kernel(flash_attention_v1 kernels/v1_fused_row.cu)
add_attention_kernel(flash_attention_v2 kernels/v2/v2a_tiled_pipelined.cu)
add_attention_kernel(flash_attention_v2b kernels/v2/v2b_tiled_pipelined.cu)

add_executable(bench_flash_attention benchmarks/bench_flash_attention.cu)
target_link_libraries(bench_flash_attention PRIVATE
  flash_attention flash_attention_v0 flash_attention_v1 flash_attention_v2 flash_attention_v2b
)

add_test(NAME flash_attention_bench COMMAND bench_flash_attention)

# A focused NCU entry point. It launches only the V2 kernel so kernel-name
# filtering and launch counting are stable across benchmark changes.
add_executable(profile_attention_v2 benchmarks/profile_attention_v2.cu)
target_link_libraries(profile_attention_v2 PRIVATE flash_attention flash_attention_v2)

add_attention_kernel(ablation_b01 kernels/ablation/b_v_access/b01.cu)
add_attention_kernel(ablation_b10 kernels/ablation/b_v_access/b10.cu)
add_attention_kernel(ablation_c1_sync kernels/ablation/c_v2a_pipeline/c1_sync.cu)
add_attention_kernel(ablation_c1_async kernels/ablation/c_v2a_pipeline/c1_async.cu)
add_attention_kernel(ablation_v2b_single kernels/ablation/d_v2b_pipeline/v2b_single.cu)
add_executable(ablation_experiments kernels/ablation/experiments.cu)
target_link_libraries(ablation_experiments PRIVATE flash_attention_v0 flash_attention_v1 flash_attention_v2 flash_attention_v2b ablation_b01 ablation_b10 ablation_c1_sync ablation_c1_async ablation_v2b_single)

# Independent V0 copy with stage-boundary events; original V0 stays frozen.
add_attention_kernel(flash_attention_v0_profile kernels/v0_unfused_profile.cu)
add_executable(profile_attention_v0 benchmarks/profile_attention_v0.cu)
target_link_libraries(profile_attention_v0 PRIVATE flash_attention_v0_profile flash_attention_v0)
add_test(NAME v0_profile_correctness COMMAND profile_attention_v0 --check)
