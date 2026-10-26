cmake_minimum_required(VERSION 3.5)
project(neo_common2)

set(CMAKE_CXX_STANDARD 14)

## By default build with code optimization and no debug info
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

add_library(neo_common2 INTERFACE)
target_include_directories(neo_common2 INTERFACE
  "$<INSTALL_INTERFACE:include>")

include_directories(
    include
    ${Boost_INCLUDE_DIRS}
)

install(TARGETS neo_common2 EXPORT export_neo_common2)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(export_neo_common2)
ament_package()
