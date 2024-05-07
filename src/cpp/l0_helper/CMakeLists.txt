set(TARGET_NAME "l0_helper")
message(INFO "Building l0_helper CMakeLists.txt")

find_package(LevelZero)

add_library(${TARGET_NAME} OBJECT
    l0_context.cpp l0_context.hpp l0_utils.cpp l0_utils.hpp usm_frame.cpp usm_frame.hpp)

# Set the position independent code property on the target
set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against the required libraries
target_link_libraries(${TARGET_NAME}
PUBLIC
    visual_ai_include
PRIVATE
    LevelZero::LevelZero  # Link the libze_loader library
)
