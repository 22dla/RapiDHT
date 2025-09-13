add_subdirectory(
    "${CMAKE_SOURCE_DIR}/3dparty/googletest"  # исходники
    "${CMAKE_BINARY_DIR}/3dparty/googletest-build"  # папка для сборки
)
enable_testing()

# List of tests
set(TESTS
  test_fdht1d
  test_fdht2d
  test_fdht3d
  test_smoke
)

# Adding tests and linking coreht to them
foreach(TEST ${TESTS})
  add_executable(${TEST} "${TEST}.cpp")
  target_link_libraries(${TEST} PRIVATE coreht gtest_main)
  install(TARGETS ${TEST} DESTINATION bin)
endforeach()
