# Find OpenMP (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found")
else()
    message(STATUS "OpenMP not found; building without OpenMP")
endif()


# adding files from source
if(USE_CUDA)
    file(GLOB SOURCES *.cpp *.cu)
else()
    file(GLOB SOURCES *.cpp)
endif()
add_library(coreht ${SOURCES})

target_link_libraries(coreht
    PRIVATE
    ${CUDA_LIBS}
    $<$<BOOL:${MPI_FOUND}>:${MPI_CXX_LIBRARIES}>
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# installing of library
install(TARGETS coreht
        RUNTIME DESTINATION bin)


# it helped!
#export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/11.8/include/:$CPATH
#export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/11.8/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
#export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/11.8/bin/:$PATH
