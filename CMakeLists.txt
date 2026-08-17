cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project. CUDA is enabled later, on demand -- see RAPIDHT_WITH_CUDA.
project(RapiDHT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for clangd/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(RAPIDHT_WITH_CUDA "Build the CUDA/cuBLAS GPU backend (Modes::GPU)" OFF)
option(RAPIDHT_WITH_MPI  "Build the MPI-distributed 3D backend" OFF)
option(RAPIDHT_BUILD_TESTS "Build the test executables" ON)
# Off by default: enabling it makes the configure step fetch Google Benchmark,
# so the default build stays buildable without network access.
option(RAPIDHT_BUILD_BENCHMARKS "Build the benchmark executable" OFF)
option(ENABLE_PROFILING "Enable function profiling" OFF)

# Set folder to installing
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
if(RAPIDHT_WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "RapiDHT: CUDA backend ON (toolkit ${CUDAToolkit_VERSION})")

    # Deliberately left to CMake, which derives a default from nvcc itself.
    # Do not default this to "native": that resolves the architecture by
    # querying an installed GPU, so it breaks on any machine that has the
    # toolkit but no card -- build servers and CI runners included. Callers
    # who do have a GPU can still pass -DCMAKE_CUDA_ARCHITECTURES=native, or
    # an explicit list such as "80;86".
    message(STATUS "RapiDHT: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "RapiDHT: CUDA backend OFF (Modes::GPU will throw at runtime)")
endif()

if(RAPIDHT_WITH_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    message(STATUS "RapiDHT: MPI backend ON (${MPI_CXX_VERSION})")
else()
    message(STATUS "RapiDHT: MPI backend OFF (3D runs single-process)")
endif()

# ---------------------------------------------------------------------------
# Generated feature-flag header
# ---------------------------------------------------------------------------
set(RAPIDHT_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rapidht_config.h.in
    ${RAPIDHT_GENERATED_INCLUDE_DIR}/rapidht_config.h
    @ONLY
)

# Include directories
include_directories(include)
include_directories(${RAPIDHT_GENERATED_INCLUDE_DIR})

if(ENABLE_PROFILING)
    add_compile_definitions(DEBUG)
endif()

# Add subdirectories
add_subdirectory(source)

# Enable testing
enable_testing()

if(RAPIDHT_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(RAPIDHT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install targets
install(TARGETS coreht DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
install(FILES ${RAPIDHT_GENERATED_INCLUDE_DIR}/rapidht_config.h DESTINATION include)
