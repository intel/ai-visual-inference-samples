set(TARGET_NAME "xpudecoder")
message(INFO " Building XPUDecoder CMakeLists.txt")

find_package(PythonDeps)
find_package(Pybind11Deps)
find_package(LibavDeps)

set(DLPACK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../external/dlpack/include)

message(STATUS "DLPack include directory: ${DLPACK_INCLUDE_DIR}")
add_library(${TARGET_NAME} OBJECT xpu_decoder.cpp)

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
PRIVATE
    ${Python_INCLUDE_DIRS}
    ${DLPACK_INCLUDE_DIR}
    ${FFMPEG_INCLUDE_DIRS}
    /usr/include/level_zero
)

target_link_libraries(${TARGET_NAME}
PUBLIC
    pybind11::module
    visual_ai_include
PRIVATE
    ${Python_LIBRARIES}
    PkgConfig::LIBAV
    vaapi_helper
    l0_helper
    context_manager
)
