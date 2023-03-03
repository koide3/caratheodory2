cmake_minimum_required(VERSION 3.0.2)
project(caratheodory)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Library
add_library(caratheodory SHARED
  src/caratheodory.cpp
)
target_include_directories(caratheodory PUBLIC
  include
  ${EIGEN3_INCLUDE_DIR}
)

# Test code
add_executable(caratheodory_test
  src/caratheodory_test.cpp
)
target_link_libraries(caratheodory_test
  caratheodory
)