cmake_minimum_required(VERSION 3.14)
project(EmotionDetectorCpp)

set(CMAKE_CXX_STANDARD 17)

# Find Python for embedding the interpreter
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${PYTHON3_INCLUDE_DIRS})

# Add pybind11 from local directory
add_subdirectory(external/pybind11)

# Include directories for pybind11
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/include)

# Build the Python binding module
pybind11_add_module(emotion_binding bindings.cpp)

# Build the main executable
add_executable(main main.cpp)

# Link against pybind11 and Python libraries
target_link_libraries(main PRIVATE pybind11::embed Python3::Python)

# Ensure it's a console executable (important for output)
set_target_properties(main PROPERTIES WIN32_EXECUTABLE FALSE)


