cmake_minimum_required(VERSION 3.15...3.27)
project(pypopsift LANGUAGES CXX CUDA)

if (NOT SKBUILD)
  message(WARNING "This CMake file is meant to be executed using scikit-build-core. See README.md for installation instructions.")
endif()

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/popsift)

nanobind_add_module(
  pypopsift
  STABLE_ABI
  src/pypopsift/pypopsift.cpp
)

target_link_libraries(pypopsift PRIVATE popsift)

set_target_properties(pypopsift PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS pypopsift LIBRARY DESTINATION pypopsift)
install(TARGETS popsift LIBRARY DESTINATION pypopsift)
