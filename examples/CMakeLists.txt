# Standalone demo programs. They print timings and error metrics but never
# fail, so they are built to keep them compiling and deliberately not
# registered with CTest -- doing so would report a pass whatever they print.
set(RAPIDHT_EXAMPLES
    fdht1d
    fdht2d
    fdht3d
)

foreach(EXAMPLE ${RAPIDHT_EXAMPLES})
    add_executable(${EXAMPLE} "${EXAMPLE}.cpp")
    target_link_libraries(${EXAMPLE} PRIVATE coreht)
    install(TARGETS ${EXAMPLE} DESTINATION bin)
endforeach()
