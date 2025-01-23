#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge

class CostMapNode(Node):
    def __init__(self):
        super().__init__('cost_map_generator')

        # Subscriber for 2D LiDAR data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/bcr_bot/scan',
            self.scan_callback,
            10
        )

        # Publisher for the cost map (as an Image)
        self.image_publisher = self.create_publisher(Image, '/cost_map', 10)

        # Bridge to convert between OpenCV and ROS Image messages
        self.bridge = CvBridge()

        # Parameters for the cost map
        self.lidar_range_max = 10.0  # Max LiDAR range (meters, adjust as per your LiDAR specs)
        self.resolution = 500  # Resolution of the cost map (pixels)
        self.center = self.resolution // 2  # Center of the map
        self.scale = self.resolution / (2 * self.lidar_range_max)  # Pixels per meter

        # Cost map parameters
        self.obstacle_cost = 100  # Cost for obstacles
        self.inflation_radius = 20  # Inflation radius in pixels (safety margin)
        self.inflation_decay = 0.5  # Cost decay factor for inflation zone

        self.get_logger().info("CostMapNode initialized.")

    def scan_callback(self, msg: LaserScan):
        """Callback to process the LaserScan data and generate the cost map."""
        # LiDAR scan angles and ranges
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Filter out invalid ranges
        ranges[np.isnan(ranges)] = 0.0
        ranges[ranges > self.lidar_range_max] = self.lidar_range_max

        # Generate a blank cost map (free space has 0 cost)
        cost_map = np.zeros((self.resolution, self.resolution), dtype=np.uint8)

        # Calculate angles for all ranges
        angles = angle_min + np.arange(len(ranges)) * angle_increment

        # Convert polar coordinates to Cartesian (vectorized)
        x = (self.center + ranges * self.scale * np.cos(angles)).astype(int)
        y = (self.center - ranges * self.scale * np.sin(angles)).astype(int)

        # Filter out points outside the map
        valid = (x >= 0) & (x < self.resolution) & (y >= 0) & (y < self.resolution)
        x = x[valid]
        y = y[valid]

        # Mark obstacles with high cost
        cost_map[y, x] = self.obstacle_cost

        # Apply inflation around obstacles
        self.apply_inflation(cost_map, self.inflation_radius, self.inflation_decay)

        # Publish the cost map as an image
        cost_image = self.bridge.cv2_to_imgmsg(cost_map, encoding='mono8')
        self.image_publisher.publish(cost_image)

        self.get_logger().info('Published cost map')

    def apply_inflation(self, cost_map, inflation_radius, decay_factor):
        """
        Apply inflation around obstacles to create a cost gradient.
        """
        # Create a kernel for inflation
        kernel_size = 2 * inflation_radius + 1
        y, x = np.ogrid[-inflation_radius:inflation_radius + 1, -inflation_radius:inflation_radius + 1]
        kernel = np.exp(-(x**2 + y**2) / (2 * (inflation_radius / 3)**2))  # Gaussian kernel
        kernel = (kernel * decay_factor * self.obstacle_cost).astype(np.uint8)

        # Find obstacle locations
        obstacle_locations = np.argwhere(cost_map == self.obstacle_cost)

        # Apply the kernel to each obstacle location
        for (y_obs, x_obs) in obstacle_locations:
            y_min = max(y_obs - inflation_radius, 0)
            y_max = min(y_obs + inflation_radius + 1, cost_map.shape[0])
            x_min = max(x_obs - inflation_radius, 0)
            x_max = min(x_obs + inflation_radius + 1, cost_map.shape[1])

            # Apply the kernel to the region around the obstacle
            cost_map[y_min:y_max, x_min:x_max] = np.maximum(
                cost_map[y_min:y_max, x_min:x_max],
                kernel[
                    y_min - (y_obs - inflation_radius):y_max - (y_obs - inflation_radius),
                    x_min - (x_obs - inflation_radius):x_max - (x_obs - inflation_radius)
                ]
            )

def main(args=None):
    rclpy.init(args=args)
    node = CostMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
