set(TARGET_NAME "utils")

add_library(${TARGET_NAME} INTERFACE)

target_include_directories(${TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_PYLOGGER)
    find_package(Pybind11Deps)

    target_link_libraries(${TARGET_NAME} INTERFACE pybind11::module)
endif(ENABLE_PYLOGGER)

if (ENABLE_PROFILING_ITT)
    target_link_libraries(${TARGET_NAME} INTERFACE ittnotify)
endif(ENABLE_PROFILING_ITT)
