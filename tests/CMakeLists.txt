# Keep GoogleTest out of `install`: without this its own install rules ship
# gtest headers and libraries alongside RapiDHT's.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${CMAKE_SOURCE_DIR}/3dparty/googletest"        # sources
    "${CMAKE_BINARY_DIR}/3dparty/googletest-build"  # build directory
    EXCLUDE_FROM_ALL
)

include(GoogleTest)

# Every TEST() is registered with CTest individually via gtest_discover_tests,
# so skipped GPU cases stay visible in the report.
set(RAPIDHT_TEST_SUITES
    test_smoke
    test_correctness
)

foreach(SUITE ${RAPIDHT_TEST_SUITES})
    add_executable(${SUITE} "${SUITE}.cpp")
    target_link_libraries(${SUITE} PRIVATE coreht gtest_main rapidht_warnings)
    gtest_discover_tests(${SUITE})
    install(TARGETS ${SUITE} DESTINATION bin)
endforeach()
