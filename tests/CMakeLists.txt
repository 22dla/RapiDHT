# List of tests
set(TESTS
  test_fdht1d
  test_fdht2d
  test_fdht3d
  test_fdhtnd
)

# Adding tests and linking coreht to them
foreach(TEST ${TESTS})
  add_executable(${TEST} "${TEST}.cpp")
  target_link_libraries(${TEST} PRIVATE coreht)
  target_link_libraries(${TEST} PRIVATE -lrt)
  install(TARGETS ${TEST} DESTINATION bin)
endforeach()
