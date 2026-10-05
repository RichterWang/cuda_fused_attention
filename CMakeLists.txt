cmake_minimum_required(VERSION 3.22)

project(cuda_fused_attention LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build for")

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
add_attention_kernel(flash_attention_v2 kernels/v2/v2_tiled_pipelined.cu)

add_executable(bench_flash_attention benchmarks/bench_flash_attention.cu)
target_link_libraries(bench_flash_attention PRIVATE
  flash_attention flash_attention_v0 flash_attention_v1 flash_attention_v2
)

add_test(NAME flash_attention_bench COMMAND bench_flash_attention)

# A focused NCU entry point. It launches only the V2 kernel so kernel-name
# filtering and launch counting are stable across benchmark changes.
add_executable(profile_attention_v2 benchmarks/profile_attention_v2.cu)
target_link_libraries(profile_attention_v2 PRIVATE flash_attention flash_attention_v2)
