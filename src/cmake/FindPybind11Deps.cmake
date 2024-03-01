# FindPybind11Deps.cmake

# Guard variable to prevent multiple inclusion effects
if(NOT FIND_PYBIND11_DEPS_INCLUDED)
    set(FIND_PYBIND11_DEPS_INCLUDED TRUE)

    find_package(Python REQUIRED COMPONENTS Interpreter Development)

    execute_process(COMMAND ${Python_EXECUTABLE} "-c"
      "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE PYBIND_PREFIX_PATH
      RESULT_VARIABLE PYBIND_PREFIX_PATH_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
    message(STATUS "PYBIND_PREFIX_PATH: ${PYBIND_PREFIX_PATH}")

    list(APPEND CMAKE_PREFIX_PATH ${PYBIND_PREFIX_PATH})
    find_package(pybind11 REQUIRED)
endif()
