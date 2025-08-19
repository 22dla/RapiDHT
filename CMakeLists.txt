cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project
project(RapiDHT LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Set folder to installing
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

# Include directories
include_directories(include)

# Options
option(USE_CUDA "Enable CUDA support" OFF)

# Find CUDA if requested
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        add_compile_definitions(USE_CUDA)
    else()
        message(WARNING "USE_CUDA=ON but CUDA toolkit not found; building without CUDA")
        set(USE_CUDA OFF)
    endif()
endif()

# Find MPI (optional)
find_package(MPI)
if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    add_compile_definitions(HAVE_MPI)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
else()
    message(STATUS "MPI not found; building without MPI support")
endif()

# Add CUDA libraries
if(USE_CUDA)
    set(CUDA_LIBS cublas cudart)  # cuBLAS + CUDA Runtime
else()
    set(CUDA_LIBS "")
endif()

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

# Install targets
install(TARGETS coreht DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
