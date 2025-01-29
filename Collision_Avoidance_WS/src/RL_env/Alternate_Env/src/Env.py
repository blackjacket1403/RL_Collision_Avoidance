import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
import math
import time
import gymnasium as gym
from gymnasium import spaces
import random
import tf2_ros
import tf_transformations

class Turtlebot_dwa(gym.Env):
    metadata={"render_modes": []}
    
    def __init__(self):
        rclpy.init()
        super().__init__()
        self.node = rclpy.create_node("turtlebot3_dwa_rl_env")
        
        self.vel_pub = self.node.create_publisher(Twist, "/cmd/vel", 10)
        self.scan_sub = self.node.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        # Publisher for Euler angles (Roll, Pitch, Yaw)
        self.euler_pub = self.node.create_publisher(Twist, "/euler_angles", 10)

        self.reset_simulation_client = self.node.create_client(Empty, '/reset_simulation')
        self.reset_world_client = self.node.create_client(Empty, '/reset_world')
        self.laser_scan = np.ones(360) * 10.0
        self.position = np.array([0.0, 0.0])
        self.orientation = 0, 0
        self.velocity = [0.0, 0.0]  # Linear and angular

        self.goal = np.array([9.0, 4.0])
        self.k = 10  # discretization of velocity
        self.n = 5   # history
        self.dt = 0.1
        
        self.max_linear_vel = 1
        self.max_angular_vel = 1
        
        self.reward_weights = {
            "goal": 2000,
            "collision": -2000,
            "proximity": -10,
            "spatial": 25,
            "r_danger_collision": 30
        }
        
        obs_shape = (self.k**2, self.n, 4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(self.k**2)
        
        self.velocity_history = []
        self.angular_velocity_history = []
        self.obstacle_history = []
        self.goal_alignment_history = []

    def scan_callback(self, msg):
        self.laser_scan = np.array(msg.ranges)
        self.laser_scan[self.laser_scan == 0] = 10.0 
    
    def odom_callback(self, msg):
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation_q = msg.pose.pose.orientation
        self.orientation = self.quaternion_to_euler(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        self.velocity = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        euler_msg = Twist()
        euler_msg.linear.x = self.orientation[0]  
        euler_msg.linear.y = self.orientation[1]  
        euler_msg.angular.z = self.orientation[2]  
        self.euler_pub.publish(euler_msg)

    def quaternion_to_euler(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        Using tf_transformations library
        """
        quaternion = [x, y, z, w]
        euler = tf_transformations.euler_from_quaternion(quaternion)
        return euler

    