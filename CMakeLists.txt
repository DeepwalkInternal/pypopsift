cmake_minimum_required(VERSION 3.15...3.27)
project(pypopsift LANGUAGES CXX CUDA)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/popsift)

nanobind_add_module(pypopsift src/pypopsift.cpp)

target_link_libraries(pypopsift PRIVATE popsift)
