add_executable(task1_app src/task1.cpp)
set_target_properties(task1_app PROPERTIES OUTPUT_NAME task1)

target_include_directories(task1_app PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(task1_app PRIVATE ${PCL_DEFINITIONS})
target_compile_definitions(task1_app PRIVATE TASKS_ROOT_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(task1_app PRIVATE
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
