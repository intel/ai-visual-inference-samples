set(TARGET_NAME "frametransform")

find_package(Pybind11Deps)

set(DLPACK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../external/dlpack/include)

message(STATUS "DLPack include directory: ${DLPACK_INCLUDE_DIR}")
add_library(${TARGET_NAME} OBJECT frame_transform.cpp)

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
PRIVATE
    ${DLPACK_INCLUDE_DIR}
)

target_link_libraries(${TARGET_NAME}
PUBLIC
    pybind11::module
    visual_ai_include
PRIVATE
    vaapi_helper
    l0_helper
    context_manager
)
