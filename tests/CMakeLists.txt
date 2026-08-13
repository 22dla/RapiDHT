add_subdirectory(
    "${CMAKE_SOURCE_DIR}/3dparty/googletest"        # sources
    "${CMAKE_BINARY_DIR}/3dparty/googletest-build"  # build directory
)

include(GoogleTest)

# GoogleTest-based suites. Every TEST() is registered with CTest individually
# via gtest_discover_tests, so skipped GPU cases stay visible in the report.
set(RAPIDHT_TEST_SUITES
    test_smoke
    test_correctness
)

foreach(SUITE ${RAPIDHT_TEST_SUITES})
    add_executable(${SUITE} "${SUITE}.cpp")
    target_link_libraries(${SUITE} PRIVATE coreht gtest_main)
    gtest_discover_tests(${SUITE})
    install(TARGETS ${SUITE} DESTINATION bin)
endforeach()

# Standalone demo programs. They print timings and error metrics but never fail,
# so they are built (to keep them compiling) and deliberately not registered as
# tests -- doing so would report a pass regardless of the numbers they produce.
set(RAPIDHT_EXAMPLES
    test_fdht1d
    test_fdht2d
    test_fdht3d
)

foreach(EXAMPLE ${RAPIDHT_EXAMPLES})
    add_executable(${EXAMPLE} "${EXAMPLE}.cpp")
    target_link_libraries(${EXAMPLE} PRIVATE coreht)
    install(TARGETS ${EXAMPLE} DESTINATION bin)
endforeach()
