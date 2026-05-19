cmake_minimum_required(VERSION 3.8)
project(puzzlebot_mc2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Goal.msg"
  "msg/SemaphoreState.msg"
  "msg/TargetFeatures.msg"
  "msg/ObstacleFeatures.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Python nodes installed as executables
install(PROGRAMS
  scripts/controller.py
  scripts/path_generator.py
  scripts/cmd_vel_bridge.py
  scripts/odometry_node.py
  scripts/analyze.py
  scripts/traffic_light_detector.py
  scripts/hsv_calibrator.py
  scripts/target_detector.py
  scripts/vs_controller.py
  scripts/color_features_detector.py
  scripts/safe_vs_controller.py
  scripts/cmd_vel_to_wheels.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch files and config
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime std_msgs geometry_msgs)
ament_package()
