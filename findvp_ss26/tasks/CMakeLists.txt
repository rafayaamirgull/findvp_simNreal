cmake_minimum_required(VERSION 3.10)
project(ViewPlanningSim VERSION 1.0 LANGUAGES C CXX)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OpenCV REQUIRED)
find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 REQUIRED)

add_subdirectory(task1)
add_subdirectory(task3)
