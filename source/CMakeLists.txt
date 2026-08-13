find_package(OpenMP REQUIRED)

# Explicit source list: the CUDA translation unit must only be compiled when
# the GPU backend is requested, which file(GLOB) cannot express.
set(RAPIDHT_SOURCES
    rapidht.cpp
    utilities.cpp
)

if(RAPIDHT_WITH_CUDA)
    list(APPEND RAPIDHT_SOURCES kernel.cu)
endif()

add_library(coreht ${RAPIDHT_SOURCES})

# PUBLIC: OpenMP pragmas and (for now) dev_array.h leak into the public headers.
target_link_libraries(coreht PUBLIC OpenMP::OpenMP_CXX)

if(RAPIDHT_WITH_CUDA)
    target_link_libraries(coreht PUBLIC CUDA::cudart CUDA::cublas)
    set_target_properties(coreht PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(RAPIDHT_WITH_MPI)
    target_link_libraries(coreht PUBLIC MPI::MPI_CXX)
endif()

# installing of library
install(TARGETS coreht
        RUNTIME DESTINATION bin)
