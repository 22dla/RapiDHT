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

# PUBLIC: the OpenMP pragmas are in the headers as well as the sources.
target_link_libraries(coreht PUBLIC OpenMP::OpenMP_CXX)

if(RAPIDHT_WITH_CUDA)
    # PRIVATE: the device state hides behind HartleyTransform::Impl, so no
    # installed header names a CUDA type and consumers need no toolkit.
    target_link_libraries(coreht PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(coreht PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(RAPIDHT_WITH_MPI)
    target_link_libraries(coreht PUBLIC MPI::MPI_CXX)
endif()

install(TARGETS coreht
        RUNTIME DESTINATION bin)
