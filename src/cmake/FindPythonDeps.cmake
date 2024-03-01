# FindPythonDeps.cmake

# Guard variable to prevent multiple inclusion effects
if(NOT FIND_PYTHON_DEPS_INCLUDED)
    set(FIND_PYTHON_DEPS_INCLUDED TRUE)

    find_package(Python COMPONENTS Interpreter Development REQUIRED)
endif()