find_package(OpenMP REQUIRED)

# Explicit source list: the CUDA translation units must only be compiled when
# the GPU backend is requested, which file(GLOB) cannot express.
set(RAPIDHT_SOURCES
    transform.cpp
    backend_cpu.cpp
    device_volume.cpp
    utilities.cpp
)

if(RAPIDHT_WITH_CUDA)
    # backend_cuda.cpp drives cuBLAS and the runtime API but launches nothing
    # itself, so it needs no device compiler; only kernels.cu does.
    list(APPEND RAPIDHT_SOURCES backend_cuda.cpp kernels.cu)
endif()

add_library(coreht ${RAPIDHT_SOURCES})

# PRIVATE: makes the internal headers reachable as "internal/..." from every
# file in the library, wherever it sits. Relying on quoted includes resolving
# relative to the including file instead only works one directory level deep:
# it holds for src/transform.cpp but not for src/internal/impl.h, which sees
# its own directory first and would need "device_array.h". Private, so nothing
# outside the library can reach these headers at all.
target_include_directories(coreht PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# PUBLIC: the OpenMP pragmas are in the headers as well as the sources.
target_link_libraries(coreht PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(coreht PRIVATE rapidht_warnings)

if(RAPIDHT_WITH_CUDA)
    # PRIVATE: the device state hides behind HartleyTransform::Impl, so no
    # installed header names a CUDA type and consumers need no toolkit.
    target_link_libraries(coreht PRIVATE CUDA::cudart CUDA::cublas)

    # Separable compilation used to be on. It is only needed when device code
    # in one translation unit calls device code in another, and there is a
    # single .cu file here, so it bought nothing -- while adding a device link
    # step to every consumer of the library. That step hands nvlink the host
    # libraries too, which is where the "Skipping incompatible libdl.a" warnings
    # came from, three per executable linked.
    set_target_properties(coreht PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(RAPIDHT_WITH_MPI)
    target_link_libraries(coreht PUBLIC MPI::MPI_CXX)
endif()

install(TARGETS coreht
        RUNTIME DESTINATION bin)
