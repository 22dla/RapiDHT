cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# GPU using option
option(RAPI_DHT_USE_GPU "Enable GPU support (requires CUDA)" ON)

# Languages
if(RAPI_DHT_USE_GPU)
    project(RapiDHT LANGUAGES CXX CUDA)
else()
    project(RapiDHT LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)

# Set folder to installing
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

# including includes and adding projects
include_directories(include)
add_subdirectory(source)
add_subdirectory(tests)

# setting tests
enable_testing()
add_test(NAME test1 COMMAND test1)
add_test(NAME test2 COMMAND test2)

# Set libraries and tests
install(TARGETS coreht DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
