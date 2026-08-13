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

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
        endif()
    endif()
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

# Install targets
install(TARGETS coreht DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
install(FILES ${RAPIDHT_GENERATED_INCLUDE_DIR}/rapidht_config.h DESTINATION include)
