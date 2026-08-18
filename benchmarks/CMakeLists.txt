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

add_executable(bench_transform bench_transform.cpp)
target_link_libraries(bench_transform PRIVATE coreht benchmark::benchmark)

# The PCIe round-trip case calls the CUDA runtime directly, to separate the
# transfer from the transform. coreht keeps CUDA private, as it should, so the
# benchmark asks for it on its own account.
if(RAPIDHT_WITH_CUDA)
    target_link_libraries(bench_transform PRIVATE CUDA::cudart)
endif()

# Bare driver for nsys/ncu: one transform, in a loop, and nothing else.
add_executable(profile_gpu3d profile_gpu3d.cpp)
target_link_libraries(profile_gpu3d PRIVATE coreht)

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
