# The Python module (B-601, first part): numpy in, numpy out, CPU only.
# Built by scikit-build-core from pyproject.toml, which turns RAPIDHT_BUILD_PYTHON on.

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# NB_STATIC: nanobind's own runtime goes into the module, so nothing else needs installing.
nanobind_add_module(_rapidht NB_STATIC rapidht_ext.cpp)
target_link_libraries(_rapidht PRIVATE coreht rapidht_warnings)

# Its own component, so a wheel carries the module alone, not coreht, the headers or
# the test executables that the other install rules describe.
install(TARGETS _rapidht LIBRARY DESTINATION rapidht COMPONENT python)
