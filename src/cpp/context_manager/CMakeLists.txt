set(TARGET_NAME "context_manager")
message(STATUS "Configuring context_manager")

add_library(${TARGET_NAME} OBJECT context_manager.cpp context_manager.hpp)

# Set the position independent code property on the target
set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against the required libraries
target_link_libraries(${TARGET_NAME}
PRIVATE
    pybind11::pybind11
    vaapi_helper
    l0_helper
)
