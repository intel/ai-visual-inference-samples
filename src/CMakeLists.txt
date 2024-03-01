cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(visual_ai)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set binaries output dir
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define the path to the custom Find modules
set(CUSTOM_CMAKE_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CUSTOM_CMAKE_MODULE_DIR}")

# Include the custom find modules
find_package(PythonDeps)
find_package(Pybind11Deps)
find_package(LibavDeps)
find_package(LevelZero)
find_package(VAAPIDeps)
find_package(OpenCV REQUIRED core imgproc)

# Add subdirectories
add_subdirectory(cpp/context_manager)
add_subdirectory(cpp/l0_helper)
add_subdirectory(cpp/utils)
add_subdirectory(cpp/vaapi_helper)
add_subdirectory(cpp/visual_ai_bind)
add_subdirectory(cpp/xpu_decoder)
add_subdirectory(cpp/xpu_encoder)

# Create the shared library from object libraries
add_library(visual_ai SHARED
    $<TARGET_OBJECTS:context_manager>
    $<TARGET_OBJECTS:l0_helper>
    $<TARGET_OBJECTS:vaapi_helper>
    $<TARGET_OBJECTS:visual_ai_bind>
    $<TARGET_OBJECTS:xpudecoder>
    $<TARGET_OBJECTS:xpuencoder>
)

# Set the position independent code property on the target
set_target_properties(visual_ai PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link against the required libraries
target_link_libraries(visual_ai PRIVATE
    pybind11::module
    Python::Python  #Defined in FindPythonDeps.cmake
    PkgConfig::LIBAV
    LevelZero::LevelZero  #Defined in FindLevelZero.cmake
    ${OpenCV_LIBS} # FIXME: Maybe better to link vaapi_helper?
    # Note: vaapi_helper and l0_helper are object libraries and don't need to be linked here
)
