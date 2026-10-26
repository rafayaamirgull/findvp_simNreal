add_executable(task3_app src/task3.cpp)
set_target_properties(task3_app PROPERTIES OUTPUT_NAME task3)

target_include_directories(task3_app PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(task3_app PRIVATE ${PCL_DEFINITIONS})
target_compile_definitions(task3_app PRIVATE TASKS_ROOT_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(task3_app PRIVATE
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
