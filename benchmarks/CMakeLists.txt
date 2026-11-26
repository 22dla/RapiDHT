include(FetchContent)

# Fetched rather than vendored, and only when benchmarks are requested, so the
# default build still needs no network access.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googlebenchmark)

if(RAPIDHT_WITH_CUDA)
    # R-5 step 0.2, variant (c): the cuFFT load callbacks as LTO-IR, embedded
    # into cufft_baseline.cu as a byte array for cufftXtSetJITCallback. One
    # lto_XX per real architecture the build targets.
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
        set(_rapidht_lto_archs "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "all-major")
        set(_rapidht_lto_archs "${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR}")
    elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "all")
        set(_rapidht_lto_archs "${CMAKE_CUDA_ARCHITECTURES_ALL}")
    else()
        set(_rapidht_lto_archs "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    set(_rapidht_lto_gencode "")
    foreach(_arch IN LISTS _rapidht_lto_archs)
        string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
        list(APPEND _rapidht_lto_gencode "-gencode=arch=compute_${_arch},code=lto_${_arch}")
    endforeach()
    list(REMOVE_DUPLICATES _rapidht_lto_gencode)
    set(_rapidht_lto_ccbin "")
    if(CMAKE_CUDA_HOST_COMPILER)
        set(_rapidht_lto_ccbin "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")
    endif()
    set(_rapidht_lto_fatbin "${CMAKE_CURRENT_BINARY_DIR}/cufft_int_lto_callback.fatbin")
    set(_rapidht_lto_header "${CMAKE_CURRENT_BINARY_DIR}/generated/cufft_int_lto_fatbin.h")
    add_custom_command(
        OUTPUT "${_rapidht_lto_header}"
        COMMAND "${CMAKE_CUDA_COMPILER}" ${_rapidht_lto_ccbin} -std=c++17 ${_rapidht_lto_gencode}
                -dc -fatbin -o "${_rapidht_lto_fatbin}" "${CMAKE_CURRENT_SOURCE_DIR}/cufft_int_lto_callback.cu"
        COMMAND "${CMAKE_COMMAND}" -DINPUT=${_rapidht_lto_fatbin} -DOUTPUT=${_rapidht_lto_header}
                -DNAME=rapidht_lto_fatbin -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_binary.cmake"
        DEPENDS cufft_int_lto_callback.cu embed_binary.cmake
        COMMENT "LTO-IR of the cuFFT load callbacks (R-5)"
        VERBATIM)
    add_custom_target(rapidht_cufft_lto_fatbin DEPENDS "${_rapidht_lto_header}")

    # cufft_baseline.cu, for every target that compiles it.
    function(rapidht_use_cufft_baseline target)
        add_dependencies(${target} rapidht_cufft_lto_fatbin)
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
        target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cufft)
    endfunction()

    # Variant (b): the legacy load callback. It must be device-linked with the
    # static cuFFT, and that cuFFT must not clash with the shared one every
    # other baseline uses, so it gets a shared library of its own that keeps
    # the static cuFFT inside (--exclude-libs: none of its symbols exported)
    # and shows only MakeCufftIntCallback. Linux only, like the measurements.
    if(NOT WIN32)
        add_library(rapidht_cufft_callback SHARED cufft_int_callback.cu)
        set_target_properties(rapidht_cufft_callback PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_RUNTIME_LIBRARY Shared
            CUDA_VISIBILITY_PRESET hidden
            CXX_VISIBILITY_PRESET hidden)
        target_link_libraries(rapidht_cufft_callback PRIVATE CUDA::cudart CUDA::cufft_static CUDA::culibos)
        target_link_options(rapidht_cufft_callback PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
endif()

set(RAPIDHT_BENCH_SOURCES bench_transform.cpp)
if(RAPIDHT_WITH_CUDA)
    list(APPEND RAPIDHT_BENCH_SOURCES cufft_baseline.cu convolution_gpu.cu)
endif()

add_executable(bench_transform ${RAPIDHT_BENCH_SOURCES})
target_link_libraries(bench_transform PRIVATE coreht benchmark::benchmark rapidht_warnings)

# The PCIe round-trip case calls the CUDA runtime directly, to separate the
# transfer from the transform, and the cuFFT baseline needs cuFFT. coreht keeps
# CUDA private, as it should, so the benchmark asks for both on its own account.
if(RAPIDHT_WITH_CUDA)
    rapidht_use_cufft_baseline(bench_transform)
    # R-5 step 2.3: IntFactored/ reaches the integer path through
    # src/internal/device_access.h, which is not public API until step 6.1.
    target_include_directories(bench_transform PRIVATE "${PROJECT_SOURCE_DIR}/src")
    if(TARGET rapidht_cufft_callback)
        target_link_libraries(bench_transform PRIVATE rapidht_cufft_callback)
        target_compile_definitions(bench_transform PRIVATE RAPIDHT_BENCH_WITH_CUFFT_CALLBACK)
    endif()
endif()

# Bare driver for nsys/ncu: one transform, in a loop, and nothing else.
add_executable(profile_gpu3d profile_gpu3d.cpp)
target_link_libraries(profile_gpu3d PRIVATE coreht rapidht_warnings)
# cudaMemGetInfo, to report what the transform holds on the device.
if(RAPIDHT_WITH_CUDA)
    target_link_libraries(profile_gpu3d PRIVATE CUDA::cudart)
    # R-5 step 2.4: the integer path through src/internal/device_access.h.
    target_include_directories(profile_gpu3d PRIVATE "${PROJECT_SOURCE_DIR}/src")
endif()

# R-4 step 0.4: float error of the GPU paths and of cuFFT against the CPU in
# double. Needs cuFFT, so only with CUDA.
if(RAPIDHT_WITH_CUDA)
    add_executable(accuracy_gpu3d accuracy_gpu3d.cpp cufft_baseline.cu)
    target_link_libraries(accuracy_gpu3d PRIVATE coreht rapidht_warnings)
    rapidht_use_cufft_baseline(accuracy_gpu3d)
    # R-5: float error on uint8 / uint16 volumes against the floors of the input
    # and of a float output.
    add_executable(accuracy_integer accuracy_integer.cpp cufft_baseline.cu)
    target_link_libraries(accuracy_integer PRIVATE coreht rapidht_warnings)
    rapidht_use_cufft_baseline(accuracy_integer)
    # R-4 step 0.5: does the cuFFT baseline fit a given volume at all.
    add_executable(capacity_gpu3d capacity_gpu3d.cpp cufft_baseline.cu)
    target_link_libraries(capacity_gpu3d PRIVATE rapidht_warnings)
    rapidht_use_cufft_baseline(capacity_gpu3d)
    # R-4, toward P2+: how fast the memory moves line sets along y and z, by shape.
    add_executable(stride_probe stride_probe.cu)
    target_link_libraries(stride_probe PRIVATE CUDA::cudart)
endif()

# Optional FFTW baseline. Ubuntu's libfftw3-dev ships a pkg-config file but no
# CMake package config, so pkg-config is the reliable route.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
endif()

if(TARGET PkgConfig::FFTW3)
    target_link_libraries(bench_transform PRIVATE PkgConfig::FFTW3)
    target_compile_definitions(bench_transform PRIVATE RAPIDHT_BENCH_WITH_FFTW)
    message(STATUS "RapiDHT: benchmarks will include the FFTW baseline")
else()
    message(STATUS "RapiDHT: FFTW not found, benchmarks will skip the baseline")
endif()
