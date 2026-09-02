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
option(RAPIDHT_WARNINGS_AS_ERRORS "Turn compiler warnings into errors" OFF)

# Set folder to installing
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
if(RAPIDHT_WITH_CUDA)
    # Must come before enable_language(CUDA), for two reasons. CMake fills
    # CMAKE_CUDA_ARCHITECTURES in with a default there, so afterwards there is
    # no way left to tell "the user asked for this" from "CMake guessed it";
    # and targets read the variable when they are created, which is later
    # still.
    #
    # That default is whatever nvcc considers oldest-supported -- 52 on CUDA
    # 12.4. It builds Maxwell SASS and leaves every newer card to JIT our
    # kernels from PTX at first launch, which is neither what anyone wants nor
    # visible in the build log.
    #
    # "all-major" is one real architecture per generation the toolkit supports,
    # so the result runs natively anywhere without a GPU having to be present
    # at configure time -- "native" needs one, and so breaks CI runners that
    # carry the toolkit but no card. It costs one nvcc pass per generation, on
    # a single .cu file.
    #
    # Override for a local build with -DCMAKE_CUDA_ARCHITECTURES=native, or an
    # explicit list such as "86".
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
        set(CMAKE_CUDA_ARCHITECTURES all-major)
    endif()

    # Compile the device code with the same host compiler as everything else.
    #
    # Left alone, nvcc picks its own: Ubuntu's nvidia-cuda-toolkit package ships
    # a g++ of the newest version that toolkit accepts and puts it first on
    # PATH. On a machine whose default compiler is newer -- CUDA 12.4 caps at
    # GCC 13, Ubuntu 25.10 ships GCC 15 -- that quietly builds kernels.cu with
    # one GCC and every .cpp with another, and links the two ABIs together.
    #
    # The result is not a configure error but an undefined reference at 90% of
    # the build, in the cold sections that only appear at -O2, so a Debug build
    # says nothing is wrong. `__cxa_call_terminate' was the symbol here.
    #
    # Pointing nvcc at CMAKE_CXX_COMPILER makes it refuse an unsupported
    # version up front, with a message that names the problem.
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()

    # Ask nvcc whether it will accept that host compiler, before
    # enable_language(CUDA) does it for us. It refuses with a clear enough
    # message of its own, but wrapped in a CMake compiler-identification
    # failure and a stack trace through three CMake modules, which buries what
    # to do about it. Probing first turns that into one paragraph.
    #
    # Only nvcc itself is looked for here, with find_program rather than
    # find_package(CUDAToolkit). CUDAToolkit does work before the language is
    # enabled, but it then has to deduce the toolkit root from nvcc's own
    # location, and on Debian and Ubuntu that gives /usr/lib/nvidia-cuda-toolkit
    # while the headers are in /usr/include -- so it fails to find
    # cuda_runtime.h and stops the configure. After enable_language(CUDA) it
    # gets the root from the compiler instead and is fine, which is where the
    # call still is.
    # The two names are not interchangeable. find_program writes to the cache,
    # and a normal variable of the same name would shadow what it found: setting
    # RAPIDHT_NVCC to an empty CMAKE_CUDA_COMPILER first and then searching into
    # it left the empty normal variable in place, so the probe below was silently
    # skipped and this whole block did nothing.
    if(CMAKE_CUDA_COMPILER)
        set(RAPIDHT_NVCC "${CMAKE_CUDA_COMPILER}")
    else()
        find_program(RAPIDHT_NVCC_PROGRAM NAMES nvcc
                     HINTS ENV CUDA_PATH ENV CUDA_HOME ENV CUDA_TOOLKIT_ROOT_DIR
                     PATH_SUFFIXES bin)
        set(RAPIDHT_NVCC "${RAPIDHT_NVCC_PROGRAM}")
    endif()

    # No nvcc to ask means no probe. enable_language(CUDA) below will report a
    # missing compiler perfectly well on its own.
    if(RAPIDHT_NVCC AND EXISTS "${RAPIDHT_NVCC}")
        set(RAPIDHT_CUDA_PROBE ${CMAKE_CURRENT_BINARY_DIR}/rapidht_host_compiler_probe.cu)
        file(WRITE ${RAPIDHT_CUDA_PROBE} "#include <cuda_runtime.h>\nint main() { return 0; }\n")
        execute_process(
            COMMAND ${RAPIDHT_NVCC}
                    -ccbin ${CMAKE_CUDA_HOST_COMPILER} -E ${RAPIDHT_CUDA_PROBE}
            RESULT_VARIABLE RAPIDHT_CUDA_PROBE_RESULT
            OUTPUT_QUIET
            ERROR_VARIABLE RAPIDHT_CUDA_PROBE_ERROR
        )
        if(NOT RAPIDHT_CUDA_PROBE_RESULT EQUAL 0)
            message(FATAL_ERROR
                "This CUDA toolkit will not accept the C++ compiler this build "
                "uses.\n"
                "  C++ compiler: ${CMAKE_CXX_COMPILER} "
                "(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})\n"
                "  nvcc:         ${RAPIDHT_NVCC}\n"
                "Every toolkit caps the host compiler version it supports. Letting "
                "nvcc fall back to one it likes is not a fix: the distribution's "
                "CUDA package ships its own compiler and puts it first on PATH, so "
                "the device code would be built with one and every .cpp with "
                "another, and the two ABIs linked together. That fails later, at "
                "link time, in optimised builds only.\n"
                "Either build everything with a compiler this toolkit accepts:\n"
                "  cmake -S . -B build -DRAPIDHT_WITH_CUDA=ON "
                "-DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13\n"
                "or install a newer toolkit from NVIDIA rather than the "
                "distribution package. Build without -DRAPIDHT_WITH_CUDA=ON to "
                "skip the GPU backend entirely.\n"
                "To make the choice stick -- including for IDEs, which pick a "
                "preset and pass no compiler of their own -- put it in a "
                "CMakeUserPresets.json inheriting from the cuda or bench preset. "
                "That file is per-machine and is not committed.\n"
                "nvcc reported:\n${RAPIDHT_CUDA_PROBE_ERROR}")
        endif()
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "RapiDHT: CUDA backend ON (toolkit ${CUDAToolkit_VERSION})")
    message(STATUS "RapiDHT: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "RapiDHT: CUDA host compiler = ${CMAKE_CUDA_HOST_COMPILER}")
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
# Generated under a rapidht/ directory so that it is reached the same way as
# every other public header, as <rapidht/config.h>.
set(RAPIDHT_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rapidht_config.h.in
    ${RAPIDHT_GENERATED_INCLUDE_DIR}/rapidht/config.h
    @ONLY
)

# Public headers are included as <rapidht/...>, so the search path is the
# directory above them. The internal headers are not here: src/ is added to the
# library target's own include path instead, so nothing outside it can reach
# them.
include_directories(include)
include_directories(${RAPIDHT_GENERATED_INCLUDE_DIR})

if(ENABLE_PROFILING)
    add_compile_definitions(DEBUG)
endif()

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
# An interface target rather than add_compile_options, which would also apply
# to 3dparty/googletest: warnings in vendored code are not ours to fix, and
# burying our own in that noise is how they came to be ignored in the first
# place. Every RapiDHT target links this; nothing else does.
#
# Guarded by COMPILE_LANGUAGE:CXX because these flags reach nvcc otherwise,
# which does not accept them directly -- host flags have to go through
# -Xcompiler.
add_library(rapidht_warnings INTERFACE)
if(MSVC)
    target_compile_options(rapidht_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    if(RAPIDHT_WARNINGS_AS_ERRORS)
        target_compile_options(rapidht_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
else()
    target_compile_options(rapidht_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
    if(RAPIDHT_WARNINGS_AS_ERRORS)
        target_compile_options(rapidht_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
endif()

add_subdirectory(src)

enable_testing()

if(RAPIDHT_BUILD_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif()

if(RAPIDHT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install targets
install(TARGETS coreht DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
install(FILES ${RAPIDHT_GENERATED_INCLUDE_DIR}/rapidht/config.h DESTINATION include/rapidht)
