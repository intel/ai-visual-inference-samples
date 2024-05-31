set(TARGET_NAME "xpuencoder")
message(STATUS "Configuring XPUEncoder")

find_package(PythonDeps)
find_package(Pybind11Deps)
find_package(LibavDeps)

add_library(${TARGET_NAME} OBJECT xpu_encoder.cpp)

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
PRIVATE
    ${Python_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    /usr/include/level_zero
)

target_link_libraries(${TARGET_NAME}
PUBLIC
    pybind11::module
PRIVATE
    ${Python_LIBRARIES}
    PkgConfig::LIBAV
    vaapi_helper
    l0_helper
    context_manager
)
