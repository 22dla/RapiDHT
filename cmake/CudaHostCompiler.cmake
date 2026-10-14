# Picks a C++ compiler that nvcc accepts, so that a CUDA build configures
# without the user having to name one.
#
# Included from the top-level CMakeLists.txt before project(), because that is
# the last point at which the compiler can still be chosen: project() detects
# it and caches it, and it cannot be changed afterwards.
#
# Every CUDA release caps the host compiler version it accepts, and the whole
# project -- every .cpp as well as kernels.cu -- has to be built with one and
# the same compiler (see the long comment next to CMAKE_CUDA_HOST_COMPILER in
# CMakeLists.txt for what mixing two does). On a distribution whose default
# GCC is newer than its CUDA package allows -- Ubuntu 25.10 ships GCC 15 and
# CUDA 12.4, which stops at GCC 13 -- the default compiler is therefore no
# good, and this file looks for an installed g++-N that nvcc does take.
#
# It stays out of the way whenever the choice has already been made:
#   * CMAKE_CXX_COMPILER given on the command line, in a preset, or already
#     in the cache from an earlier configure;
#   * the CXX environment variable set;
#   * CMAKE_CUDA_HOST_COMPILER set explicitly;
#   * a toolchain file in use (it chooses compilers itself);
#   * a Windows host, where the only supported host compiler is MSVC.
# If the default compiler is accepted, nothing is changed either.
#
# When nothing installed is accepted it does nothing, and the probe in
# CMakeLists.txt reports the problem with the list of what was tried.

if(NOT RAPIDHT_WITH_CUDA
   OR CMAKE_HOST_WIN32
   OR DEFINED CMAKE_CXX_COMPILER
   OR DEFINED ENV{CXX}
   OR DEFINED CMAKE_CUDA_HOST_COMPILER
   OR CMAKE_TOOLCHAIN_FILE)
    return()
endif()

# The same nvcc that CMakeLists.txt and enable_language(CUDA) will use.
if(DEFINED CMAKE_CUDA_COMPILER)
    set(_rapidht_nvcc "${CMAKE_CUDA_COMPILER}")
elseif(DEFINED ENV{CUDACXX})
    set(_rapidht_nvcc "$ENV{CUDACXX}")
else()
    find_program(_RAPIDHT_NVCC_SEARCH NAMES nvcc
                 HINTS ENV CUDA_PATH ENV CUDA_HOME ENV CUDA_TOOLKIT_ROOT_DIR
                 PATH_SUFFIXES bin)
    set(_rapidht_nvcc "${_RAPIDHT_NVCC_SEARCH}")
    unset(_RAPIDHT_NVCC_SEARCH CACHE)
endif()
if(NOT _rapidht_nvcc OR NOT EXISTS "${_rapidht_nvcc}")
    # enable_language(CUDA) reports a missing nvcc well enough on its own.
    return()
endif()

set(_rapidht_probe "${CMAKE_BINARY_DIR}/CMakeFiles/rapidht_host_compiler_select.cu")
file(WRITE "${_rapidht_probe}" "#include <cuda_runtime.h>\nint main() { return 0; }\n")

# Asks nvcc whether it accepts the given host compiler; the answer goes to
# out_var as TRUE or FALSE.
function(_rapidht_nvcc_accepts compiler out_var)
    execute_process(
        COMMAND "${_rapidht_nvcc}" -ccbin "${compiler}" -E "${_rapidht_probe}"
        RESULT_VARIABLE result
        OUTPUT_QUIET ERROR_QUIET)
    if(result EQUAL 0)
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# The compiler project() would pick on its own: c++ on PATH.
find_program(_RAPIDHT_DEFAULT_CXX NAMES c++ g++)
set(_rapidht_default_cxx "${_RAPIDHT_DEFAULT_CXX}")
unset(_RAPIDHT_DEFAULT_CXX CACHE)

if(_rapidht_default_cxx)
    _rapidht_nvcc_accepts("${_rapidht_default_cxx}" _rapidht_ok)
    if(_rapidht_ok)
        # The usual case: nothing to do.
        return()
    endif()
endif()

# Versioned GCCs, newest first: the newest one nvcc takes is the closest to
# the system default.
set(RAPIDHT_HOST_COMPILERS_TRIED "${_rapidht_default_cxx}")
foreach(_rapidht_ver RANGE 20 9 -1)
    find_program(_RAPIDHT_CANDIDATE_CXX NAMES g++-${_rapidht_ver})
    set(_rapidht_cxx "${_RAPIDHT_CANDIDATE_CXX}")
    unset(_RAPIDHT_CANDIDATE_CXX CACHE)
    if(NOT _rapidht_cxx)
        continue()
    endif()
    list(APPEND RAPIDHT_HOST_COMPILERS_TRIED "${_rapidht_cxx}")
    _rapidht_nvcc_accepts("${_rapidht_cxx}" _rapidht_ok)
    if(NOT _rapidht_ok)
        continue()
    endif()

    # The matching C compiler, so that anything built as C (googletest's
    # project enables the language) comes from the same GCC.
    find_program(_RAPIDHT_CANDIDATE_CC NAMES gcc-${_rapidht_ver})
    set(_rapidht_cc "${_RAPIDHT_CANDIDATE_CC}")
    unset(_RAPIDHT_CANDIDATE_CC CACHE)

    message(STATUS "RapiDHT: ${_rapidht_nvcc} rejects the default compiler "
                   "${_rapidht_default_cxx}; building everything with "
                   "${_rapidht_cxx} instead")
    set(CMAKE_CXX_COMPILER "${_rapidht_cxx}" CACHE FILEPATH
        "C++ compiler, chosen to match what nvcc accepts")
    if(_rapidht_cc)
        set(CMAKE_C_COMPILER "${_rapidht_cc}" CACHE FILEPATH
            "C compiler, chosen to match the C++ compiler")
    endif()
    return()
endforeach()

# Nothing installed is accepted. Leave the default in place; the probe in
# CMakeLists.txt fails with RAPIDHT_HOST_COMPILERS_TRIED in its message.
