set(TARGET_NAME "visual_ai_bind")
message(STATUS "Configuring visual_ai_bind")

find_package(Pybind11Deps)

add_library(${TARGET_NAME} OBJECT visual_ai_bind.cpp utils_bind.cpp overlay_bind.cpp)

# Set the position independent code property on the target
set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against the required libraries
target_link_libraries(${TARGET_NAME} PRIVATE
    pybind11::pybind11
    visual_ai_include
    xpudecoder
    vaapi_helper
    utils
    frametransform
)
