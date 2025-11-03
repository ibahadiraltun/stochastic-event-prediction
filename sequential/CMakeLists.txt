cmake_minimum_required(VERSION 3.14)
project(step-sequential
    VERSION 1.0.0
    DESCRIPTION "STEP: Sequential Temporal Event Predictor"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Main executable
add_executable(step-sequential src/main.cpp)
target_include_directories(step-sequential PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Scorer executable
add_executable(scorer src/scorer.cpp)
target_include_directories(scorer PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Installation
install(TARGETS step-sequential scorer DESTINATION bin)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
