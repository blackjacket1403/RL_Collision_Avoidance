import rclpy
from rclpy.node import Node
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
import math
import time
import random


class TurtleBot3DWAEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        rclpy.init()
        super().__init__()

        self.node = rclpy.create_node("turtlebot3_dwa_rl_env")

        # ROS 2 Publishers and Subscribers
        self.vel_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.node.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Gazebo service clients
        self.reset_simulation_client = self.node.create_client(Empty, '/reset_simulation')
        self.reset_world_client = self.node.create_client(Empty, '/reset_world')
        self.pause_physics_client = self.node.create_client(Empty, '/pause_physics')
        self.unpause_physics_client = self.node.create_client(Empty, '/unpause_physics')

        # Robot state
        self.laser_scan = np.ones(360) * 10.0  # Default to max range
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0  # Yaw
        self.velocity = [0.0, 0.0]  # [Linear, Angular]

        self.goal = np.array([9.0, 4.0])  # Example goal position

        # RL Parameters
        self.k = 10  # Discretization for velocity
        self.n = 5   # Temporal history
        self.dt = 0.1

        # Robot constraints
        self.max_linear_vel = 0.22
        self.max_angular_vel = 2.84
        self.acc_linear = 0.1
        self.acc_angular = 0.1

        # Reward weights
        self.reward_weights = {
            "goal": 2000,
            "collision": -2000,
            "proximity": -30,
            "spatial": 25,
        }

        # Observation and Action Space
        obs_shape = (self.k ** 2, self.n, 3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(self.k ** 2)

        # Initialize history buffers
        self.velocity_history = []
        self.angular_velocity_history = []
        self.obstacle_history = []
        self.goal_alignment_history = []

    def scan_callback(self, msg):
        """Process incoming laser scan data"""
        self.laser_scan = np.array(msg.ranges)
        self.laser_scan[self.laser_scan == 0] = 10.0  # Replace invalid readings

    def odom_callback(self, msg):
        """Process incoming odometry data"""
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation_q = msg.pose.pose.orientation
        _, _, self.orientation = self.quaternion_to_euler(
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        )
        self.velocity = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    @staticmethod
    def quaternion_to_euler(x, y, z, w):
        """Convert quaternion to Euler angles"""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    def compute_dwa_velocities(self):
        """Compute possible velocity pairs using Dynamic Window Approach"""
        v_min = max(0, self.velocity[0] - self.acc_linear * self.dt)
        v_max = min(self.max_linear_vel, self.velocity[0] + self.acc_linear * self.dt)
        w_min = max(-self.max_angular_vel, self.velocity[1] - self.acc_angular * self.dt)
        w_max = min(self.max_angular_vel, self.velocity[1] + self.acc_angular * self.dt)

        v_range = np.linspace(v_min, v_max, self.k)
        w_range = np.linspace(w_min, w_max, self.k)
        velocities = [(v, w) for v in v_range for w in w_range]
        return velocities

    def compute_reward(self, action):
        """Compute reward based on current state and action"""
        v, w = self.compute_dwa_velocities()[action]

        # Goal alignment reward
        goal_distance = np.linalg.norm(self.goal - self.position)
        goal_reward = self.reward_weights["goal"] if goal_distance < 0.3 else -goal_distance

        # Collision penalty
        collision_penalty = self.reward_weights["collision"] if min(self.laser_scan) < 0.5 else 0

        # Proximity penalty
        proximity_penalty = sum(
            [self.reward_weights["proximity"] / dist if dist < 1.0 else 0 for dist in self.laser_scan]
        )

        # Steering reward
        spatial_reward = self.reward_weights["spatial"] if w > 0 else -self.reward_weights["spatial"]

        return goal_reward + collision_penalty + proximity_penalty + spatial_reward

    def get_observation(self):
        """Get the current observation of the environment"""
        velocities = self.compute_dwa_velocities()
        
        # Initialize history buffers if empty
        if len(self.velocity_history) == 0:
            for _ in range(self.n):
                self.velocity_history.append(np.zeros(self.k ** 2))
                self.angular_velocity_history.append(np.zeros(self.k ** 2))
                self.obstacle_history.append(np.zeros(self.k ** 2))
                self.goal_alignment_history.append(np.zeros(self.k ** 2))
        
        # Compute current costs
        obstacle_costs = np.array([
            1.0 / min(self.laser_scan) if min(self.laser_scan) < 1.0 else 0
            for _ in velocities
        ])
        
        goal_costs = np.array([
            np.linalg.norm(self.goal - (self.position + np.array([v * self.dt, w * self.dt])))
            for v, w in velocities
        ])
        
        # Update history buffers
        self.velocity_history.pop(0)
        self.angular_velocity_history.pop(0)
        self.obstacle_history.pop(0)
        self.goal_alignment_history.pop(0)
        
        # Add new observations
        self.velocity_history.append(np.array([v[0] for v in velocities]))
        self.angular_velocity_history.append(np.array([v[1] for v in velocities]))
        self.obstacle_history.append(obstacle_costs)
        self.goal_alignment_history.append(goal_costs)
        
        # Convert to numpy arrays and stack
        velocity_arr = np.array(self.velocity_history)
        angular_velocity_arr = np.array(self.angular_velocity_history)
        obstacle_arr = np.array(self.obstacle_history)
        
        # Create observation with shape (k^2, n, 3)
        observation = np.stack([
            velocity_arr.T,
            angular_velocity_arr.T,
            obstacle_arr.T
        ], axis=-1)
        
        assert observation.shape == (self.k ** 2, self.n, 3), \
            f"Observation shape mismatch. Expected {(self.k ** 2, self.n, 3)}, got {observation.shape}"
        
        return observation.astype(np.float32)

    def step(self, action):
        """Execute one time step within the environment"""
        v, w = self.compute_dwa_velocities()[action]
        
        # Publish velocity command
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = w
        self.vel_pub.publish(twist_msg)

        # Allow ROS to process messages
        rclpy.spin_once(self.node, timeout_sec=self.dt)

        # Compute reward and check if done
        reward = self.compute_reward(action)
        done = np.linalg.norm(self.goal - self.position) < 0.3
        truncated = False

        return self.get_observation(), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment state"""
        super().reset(seed=seed)
        
        # Wait for services
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Reset simulation service not available, waiting...')
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Reset world service not available, waiting...')
        
        # Pause physics
        self.pause_physics_client.call_async(Empty.Request())
        
        # Reset simulation and world
        self.reset_simulation_client.call_async(Empty.Request())
        self.reset_world_client.call_async(Empty.Request())
        
        # Clear history buffers
        self.velocity_history = []
        self.angular_velocity_history = []
        self.obstacle_history = []
        self.goal_alignment_history = []
        
        # Reset robot state
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.velocity = [0.0, 0.0]
        
        # Set new random goal if specified
        if options and 'random_goal' in options and options['random_goal']:
            self.goal = np.array([
                random.uniform(-4.0, 4.0),
                random.uniform(-4.0, 4.0)
            ])
        
        # Allow time for reset
        time.sleep(0.5)
        
        # Unpause physics
        self.unpause_physics_client.call_async(Empty.Request())
        
        # Get initial observation
        observation = self.get_observation()
        
        # Ensure proper spin of ROS node
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        return observation, {}

    def shutdown(self):
        """Cleanup function to be called when the environment is closed"""
        # Stop the robot
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.vel_pub.publish(stop_msg)
        
        # Clean up ROS nodes and services
        if hasattr(self, 'reset_simulation_client'):
            self.reset_simulation_client.destroy()
        if hasattr(self, 'reset_world_client'):
            self.reset_world_client.destroy()
        if hasattr(self, 'pause_physics_client'):
            self.pause_physics_client.destroy()
        if hasattr(self, 'unpause_physics_client'):
            self.unpause_physics_client.destroy()
        
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # Example usage
    env = TurtleBot3DWAEnv()
    try:
        obs, _ = env.reset()
        for _ in range(100):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            if done:
                obs, _ = env.reset()
    finally:
        env.shutdown()