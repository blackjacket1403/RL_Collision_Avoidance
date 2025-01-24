#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge

class DensityMapNode(Node):
    def __init__(self):
        super().__init__('density_map_generator')

        # Subscriber for 2D LiDAR data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/bcr_bot/scan',
            self.scan_callback,
            10
        )

        # Publisher for the density map (as an Image)
        self.image_publisher = self.create_publisher(Image, '/density_map', 10)

        # Bridge for ROS <-> OpenCV image conversion
        self.bridge = CvBridge()

        # Map and LiDAR parameters
        self.lidar_range_max = 10.0  # Max LiDAR range (meters)
        self.resolution = 500  # Fixed resolution for the map
        self.center = self.resolution // 2  # Map center point
        self.scale = self.resolution / (2 * self.lidar_range_max)  # Pixels per meter

        # Gaussian blur parameters
        self.gaussian_kernel_size = 15  # Size of the Gaussian kernel
        self.gaussian_sigma = 5  # Standard deviation for Gaussian blur

        self.get_logger().info("DensityMapNode initialized.")

    def scan_callback(self, msg: LaserScan):
        """Process LaserScan data and generate a density map."""
        # Extract ranges and angles
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # Filter invalid ranges
        valid = (ranges > 0) & (ranges <= self.lidar_range_max)
        ranges = ranges[valid]
        angles = angles[valid]

        # Convert polar coordinates to Cartesian and map to grid
        x = (self.center + ranges * self.scale * np.cos(angles)).astype(int)
        y = (self.center - ranges * self.scale * np.sin(angles)).astype(int)

        # Initialize a blank density map (black image)
        density_map = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        # Mark obstacle points with high density
        density_map[y, x] = 1.0  # Obstacle points have maximum density

        # Mark occluded regions behind obstacles as high density
        self.mark_occluded_regions(density_map, x, y)

        # Apply Gaussian blur to create smooth transitions
        density_map = cv2.GaussianBlur(density_map, (self.gaussian_kernel_size, self.gaussian_kernel_size), self.gaussian_sigma)

        # Normalize the density map to [0, 255]
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        density_map = density_map.astype(np.uint8)

        # Publish the density map as a ROS Image
        density_image = self.bridge.cv2_to_imgmsg(density_map, encoding='mono8')
        self.image_publisher.publish(density_image)

        self.get_logger().info('Published density map')

    def mark_occluded_regions(self, density_map, x, y):
        """Mark regions behind obstacles as high density."""
        # Create a grid of coordinates
        y_grid, x_grid = np.indices(density_map.shape)

        # Compute the direction vectors from the LiDAR origin to each obstacle point
        dx = x - self.center
        dy = y - self.center

        # Compute the distance from the LiDAR origin to each obstacle point
        dist = np.sqrt(dx**2 + dy**2)

        # Normalize the direction vectors
        dx_norm = dx / dist
        dy_norm = dy / dist

        # Compute the distance from the LiDAR origin to each grid point
        dist_grid = np.sqrt((x_grid - self.center)**2 + (y_grid - self.center)**2)

        # Compute the projection of each grid point onto the direction vectors
        proj = (x_grid - self.center) * dx_norm[:, None, None] + (y_grid - self.center) * dy_norm[:, None, None]

        # Mark occluded regions (behind obstacles)
        occluded = (proj > dist[:, None, None]) & (np.abs(proj - dist[:, None, None]) < 1.0)
        density_map[occluded.any(axis=0)] = 1.0

def main(args=None):
    rclpy.init(args=args)
    node = DensityMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
