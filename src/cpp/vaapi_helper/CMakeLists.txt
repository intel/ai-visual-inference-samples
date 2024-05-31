# ==============================================================================
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier:
# ==============================================================================

set(TARGET_NAME "vaapi_helper")
message(STATUS "Configuring vaapi_helper")

find_package(VAAPIDeps)
find_package(OpenCV REQUIRED core imgproc)

add_library(${TARGET_NAME} OBJECT
    vaapi_context.cpp vaapi_frame_pool.cpp vaapi_frame.cpp vaapi_frame_converter.cpp vaapi_utils.cpp
    vaapi_context.hpp vaapi_frame_pool.hpp vaapi_frame.hpp vaapi_frame_converter.hpp vaapi_utils.hpp
    vaapi_overlay.cpp
    vaapi_overlay.hpp)

target_include_directories(${TARGET_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${VA_INCLUDE_DIRS}
PRIVATE
    ${VADRM_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME}
PUBLIC
    ${VA_LIBRARIES}
    utils
    visual_ai_include
PRIVATE
    ${OpenCV_LIBS}
    ${VADRM_LIBRARIES}
)
