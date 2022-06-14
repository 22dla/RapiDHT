###########################
# Build project together
###########################

# macro(init_path)
    # set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    # set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin) # exe
    # set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib) # library
    # set(ENGINE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)
    # set(TASKFLOW_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/taskflow)
# endmacro(init_path)

macro(init_variables)
    set (PROJECT_LIB_FILES "")
    set (PROJECT_INCLUDE_DIRS "")
    set (TARGET_LIB_FILES "")
    set (TARGET_INCLUDE_DIRS "")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fp:fast /Ox")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /Ox")
endmacro(init_variables)

macro(init_openmp)
    option(USE_OPENMP "Use openmp" ON)
    if(USE_OPENMP)
        add_definitions(-DPE_USE_OMP)
        find_package(OpenMP)
        if(APPLE)
            execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_INCLUDE_DIR "${BREW_LIBOMP_PREFIX}/include")
            message(STATUS "Using Homebrew libomp from ${BREW_LIBOMP_PREFIX}")
            include_directories("${OpenMP_INCLUDE_DIR}")
            # set (LIB_FILES ${OpenMP_omp_LIBRARY})
            list(APPEND PROJECT_LIB_FILES ${OpenMP_omp_LIBRARY})
        endif()
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endmacro(init_openmp)

macro(use_taskflow)
    list(APPEND TARGET_INCLUDE_DIRS ${TASKFLOW_INCLUDE_DIR})
endmacro(use_taskflow)

macro(include_and_link)
    include_directories(${PROJECT_INCLUDE_DIRS})
    include_directories(${TARGET_INCLUDE_DIRS})
    link_libraries(${PROJECT_LIB_FILES})
    link_libraries(${TARGET_LIB_FILES})
endmacro(include_and_link)

###########################
# Build project separately
###########################

macro(init_project proj_name)
    SET(CMAKE_CXX_STANDARD 17)
    add_definitions(-DGLOBALBENCHMARK)
    set(CMAKE_MODULE_PATH ${ENGINE_PATH}/cmake)
    set(ENGINE_INCLUDE_DIR ${ENGINE_PATH}/src)
    set(TASKFLOW_INCLUDE_DIR ${ENGINE_PATH}/3rdparty/taskflow)
    include_directories(${ENGINE_INCLUDE_DIR})
    include_directories(${TASKFLOW_INCLUDE_DIR})
    option(ENABLE_EXTENDED_ALIGNED_STORAGE "Enable extended aligned storage" ON)
    if (ENABLE_EXTENDED_ALIGNED_STORAGE)
        add_compile_definitions("_ENABLE_EXTENDED_ALIGNED_STORAGE")
    else()
        add_compile_definitions("_DISABLE_EXTENDED_ALIGNED_STORAGE")
    endif()
	option(USE_CUDA "Use cuda" ON)
    if(NOT_DEFINED_CMAKE_CUDA_ARCHITECTURE)
        set(CMAKE_CUDA_ARCHITECTURES 52 61 70 72 75 CACHE STRING "CUDA architectures" FORCE) 
    endif() 
endmacro()

macro(add_engine_module module lib)
    add_subdirectory(${ENGINE_PATH}/src/${module} ${CMAKE_CURRENT_BINARY_DIR}/${module})
    link_libraries(${lib})
endmacro()

macro(use_taskflow_local)
    include_directories(${ENGINE_PATH}/3rdparty/taskflow)
endmacro()

macro(use_openmp_local)
    option(USE_OPENMP "Use openmp" ON)
    if(USE_OPENMP)
        add_definitions(-DPE_USE_OMP)
        find_package(OpenMP)
        if(APPLE)
            execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_INCLUDE_DIR "${BREW_LIBOMP_PREFIX}/include")
            message(STATUS "Using Homebrew libomp from ${BREW_LIBOMP_PREFIX}")
            include_directories("${OpenMP_INCLUDE_DIR}")
            # set (LIB_FILES ${OpenMP_omp_LIBRARY})
            list(APPEND PROJECT_LIB_FILES ${OpenMP_omp_LIBRARY})
        endif()
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endmacro()

macro(use_cuda_local target_name)
    option(USE_CUDA "Use cuda" ON)
    if (USE_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        target_include_directories(${target_name} PUBLIC ${ENGINE_PATH}/3rdparty/cuda/common/inc)
        link_directories(${CUDAToolkit_LIBRARY_DIR})
        target_link_libraries(${target_name} PRIVATE CUDA::cudart CUDA::cuda_driver)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 61)
        set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -rdc=true -prec-div=false -prec-sqrt=false -ftz=true -use_fast_math)
        add_definitions(-DPE_USE_CUDA)
    endif()
endmacro()

macro(use_freeglut_local target_name)
    if(WIN32)
        set(freeglut_path ${ENGINE_PATH}/3rdparty/freeglut)
        set(freeglut_src_path ${freeglut_path}/include)
        set(freeglut_lib_path ${freeglut_path}/lib/x64)
        include_directories(${freeglut_src_path})
        set(glew_libs ${freeglut_lib_path}/glew32.lib)
        list(APPEND freeglut_libs ${glew_libs})
        set(glut_libs debug ${freeglut_lib_path}/freeglut.lib optimized ${freeglut_lib_path}/freeglut.lib)
        list(APPEND freeglut_libs ${glut_libs})
    else()
        message(STATUS "Unix/MacOS unsupported yet")
    endif()
    # link_libraries(${freeglut_libs})
    target_link_libraries(${target_name} PRIVATE ${freeglut_libs})
endmacro()

macro(add_exe exe_name)
	#project src files
	file(GLOB_RECURSE cpp_files ${PROJECT_PATH}/*.cpp)
	file(GLOB_RECURSE h_files ${PROJECT_PATH}/*.h)
    if(USE_CUDA)
		file(GLOB_RECURSE cu_files ${PROJECT_PATH}/*.cu)
		file(GLOB_RECURSE cuh_files ${PROJECT_PATH}/*.cuh)
	else()
		set(cu_files "")
        set(cuh_files "")
    endif(USE_CUDA)
    list(APPEND src_files ${h_files} ${cpp_files} ${cu_files} ${cuh_files})
    message(STATUS "add exe " ${exe_name} " with files " ${src_files})
    add_executable(${exe_name} ${src_files})
endmacro()

macro(add_lib lib_name src_path)
	#project src files
	file(GLOB_RECURSE cpp_files ${src_path}/*.cpp)
	file(GLOB_RECURSE h_files ${src_path}/*.h)
    if(USE_CUDA)
		file(GLOB_RECURSE cu_files ${src_path}/*.cu)
        file(GLOB_RECURSE cuh_files ${src_path}/*.cuh)
        message(STATUS "Use cuda in lib " ${lib_name})
	else()
		set(cu_files "")
        set(cuh_files "")
        set_target_properties(${lib_name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif(USE_CUDA)
    list(APPEND src_files ${h_files} ${cpp_files} ${cu_files} ${cuh_files})
    message(STATUS "add lib " ${lib_name} " with files " ${src_files})
    add_library(${lib_name} ${src_files})
endmacro()
