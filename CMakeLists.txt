cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project
project(RapiDHT LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# Set folder to installing
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

# Include directories
include_directories(include)

# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA found: ${CUDA_VERSION}")

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Add CUDA libraries
set(CUDA_LIBS cublas cudart)  # cuBLAS + CUDA Runtime

# Profiler option
option(ENABLE_PROFILING "Enable function profiling" OFF)

if(ENABLE_PROFILING)
    add_compile_definitions(DEBUG)
endif()

# Add subdirectories
add_subdirectory(source)
add_subdirectory(tests)

# Enable testing
enable_testing()
add_test(NAME test1 COMMAND test1)
add_test(NAME test2 COMMAND test2)

# Install targets
install(TARGETS coreht DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
