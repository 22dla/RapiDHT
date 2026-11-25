# Writes a binary file into a C++ header as a byte array, at build time:
#
#   cmake -DINPUT=<file> -DOUTPUT=<header> -DNAME=<array> -P embed_binary.cmake
#
# Used for the LTO-IR fatbin of the cuFFT load callbacks (R-5 step 0.2), which
# cufftXtSetJITCallback takes as a pointer and a size. bin2c would do the same
# but writes only to standard output, which add_custom_command cannot redirect
# portably.

file(READ "${INPUT}" _hex HEX)
string(LENGTH "${_hex}" _length)
math(EXPR _bytes "${_length} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _array "${_hex}")
# A line break every 16 bytes keeps the header readable in an editor.
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){16})" "\\1\n    " _array "${_array}")
file(WRITE "${OUTPUT}"
    "// Generated from ${INPUT} by benchmarks/embed_binary.cmake; do not edit.\n"
    "#pragma once\n"
    "// ${_bytes} bytes; a fatbin is read in 8-byte words.\n"
    "alignas(8) static const unsigned char ${NAME}[] = {\n    ${_array}\n};\n")
