cmake_minimum_required(VERSION 3.8)
project(my_env)

# Set compiler warnings for C++ (optional, if you have C++ code)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)  # Required for Python nodes
find_package(gazebo_ros REQUIRED)  # Required for Gazebo integration

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install URDF files
install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/
)
# Install world files
install(
  DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}/
)

# Enable testing (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip copyright and cpplint checks (uncomment if needed)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Finalize the package
ament_package()