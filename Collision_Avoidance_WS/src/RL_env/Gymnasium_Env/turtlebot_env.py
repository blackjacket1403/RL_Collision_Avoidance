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

        # Initialize ROS node and subscribers/publishers
        self.node = rclpy.create_node("turtlebot3_dwa_rl_env")
        self.vel_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.node.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Initialize Gazebo service clients
        self.reset_simulation_client = self.node.create_client(Empty, '/reset_simulation')
        self.reset_world_client = self.node.create_client(Empty, '/reset_world')
        self.pause_physics_client = self.node.create_client(Empty, '/pause_physics')
        self.unpause_physics_client = self.node.create_client(Empty, '/unpause_physics')

        # Environment parameters
        self.k = 20  # Velocity discretization
        self.n = 10   # History length
        self.dt = 0.1  # Time step
        self.collision_threshold = 0.2  # Meters
        self.goal_threshold = 0.3  # Meters
        self.max_episode_steps = 10000

        # Robot parameters
        self.max_linear_vel = 0.22
        self.max_angular_vel = 2.84
        self.acc_linear = 0.1
        self.acc_angular = 0.1

        # State variables
        self.laser_scan = np.ones(360) * 10.0
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.velocity = [0.0, 0.0]
        self.goal = np.array([9.0, 4.0])
        self.steps_taken = 0

        # History buffers
        self.velocity_history = []
        self.angular_velocity_history = []
        self.obstacle_history = []
        self.goal_alignment_history = []

        # Reward weights
        self.reward_weights = {
            "goal": 2000.0,
            "collision": -2000.0,
            "proximity": -30.0,
            "spatial": 25.0,
            "progress": 100.0,
            "time_penalty": -1.0
        }

        # Gym spaces
        obs_shape = (self.k ** 2, self.n, 3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.k ** 2)

        # Initialize previous distance for progress reward
        self.previous_distance = None

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.laser_scan = np.array(msg.ranges)
        self.laser_scan[self.laser_scan == 0] = 10.0  # Replace invalid readings

    def odom_callback(self, msg):
        """Process odometry data"""
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])
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
        """Compute velocity pairs using Dynamic Window Approach"""
        v_min = max(0, self.velocity[0] - self.acc_linear * self.dt)
        v_max = min(self.max_linear_vel, self.velocity[0] + self.acc_linear * self.dt)
        w_min = max(-self.max_angular_vel, self.velocity[1] - self.acc_angular * self.dt)
        w_max = min(self.max_angular_vel, self.velocity[1] + self.acc_angular * self.dt)

        v_range = np.linspace(v_min, v_max, self.k)
        w_range = np.linspace(w_min, w_max, self.k)
        return [(v, w) for v in v_range for w in w_range]

    def compute_reward(self, action):
        """Compute reward for current state-action pair"""
        # Calculate current distance to goal
        current_distance = np.linalg.norm(self.goal - self.position)
        
        # Initialize reward components
        goal_reward = 0
        progress_reward = 0
        time_penalty = self.reward_weights["time_penalty"]
        
        # Goal reward
        if current_distance < self.goal_threshold:
            goal_reward = self.reward_weights["goal"]
            
        # Progress reward
        if self.previous_distance is not None:
            progress = self.previous_distance - current_distance
            progress_reward = self.reward_weights["progress"] * progress
            
        # Update previous distance
        self.previous_distance = current_distance
        
        # Proximity penalty
        proximity_penalty = sum(
            [self.reward_weights["proximity"] / dist 
             for dist in self.laser_scan if self.collision_threshold < dist < 1.0]
        )
        
        return goal_reward + progress_reward + proximity_penalty + time_penalty

    def get_observation(self):
        """Get current observation"""
        velocities = self.compute_dwa_velocities()
        
        # Initialize or check history buffers
        if (len(self.velocity_history) != self.n or 
            len(self.angular_velocity_history) != self.n or 
            len(self.obstacle_history) != self.n or 
            len(self.goal_alignment_history) != self.n):
            
            # Clear existing history if any
            self.velocity_history = []
            self.angular_velocity_history = []
            self.obstacle_history = []
            self.goal_alignment_history = []
            
            # Initialize with zeros
            for _ in range(self.n):
                self.velocity_history.append(np.zeros(self.k ** 2))
                self.angular_velocity_history.append(np.zeros(self.k ** 2))
                self.obstacle_history.append(np.zeros(self.k ** 2))
                self.goal_alignment_history.append(np.zeros(self.k ** 2))
        
        # Compute current costs
        obstacle_costs = np.array([
            1.0 / min(self.laser_scan) if min(self.laser_scan) < 0.5 else 0
            for _ in velocities
        ])
        
        goal_costs = np.array([
            np.linalg.norm(self.goal - (self.position + np.array([v * self.dt, w * self.dt])))
            for v, w in velocities
        ])
        
        # Update history buffers (remove oldest, add newest)
        self.velocity_history.pop(0)
        self.angular_velocity_history.pop(0)
        self.obstacle_history.pop(0)
        self.goal_alignment_history.pop(0)
        
        # Add new observations
        self.velocity_history.append(np.array([v[0] for v in velocities]))
        self.angular_velocity_history.append(np.array([v[1] for v in velocities]))
        self.obstacle_history.append(obstacle_costs)
        self.goal_alignment_history.append(goal_costs)
        
        # Convert to numpy arrays and ensure correct shapes
        velocity_arr = np.array(self.velocity_history)
        angular_velocity_arr = np.array(self.angular_velocity_history)
        obstacle_arr = np.array(self.obstacle_history)
        
        # Create observation with shape (k^2, n, 3)
        observation = np.stack([
            velocity_arr.T,
            angular_velocity_arr.T,
            obstacle_arr.T
        ], axis=-1)
        
        # Add shape assertion for debugging
        assert observation.shape == (self.k ** 2, self.n, 3), \
            f"Observation shape mismatch. Expected {(self.k ** 2, self.n, 3)}, got {observation.shape}"
        
        return observation.astype(np.float32)

        return observation.astype(np.float32)
    def reset_gazebo(self):
        """Reset Gazebo simulation"""
        try:
            # Stop the robot first
            stop_msg = Twist()
            self.vel_pub.publish(stop_msg)
            
            # Wait for services with timeout
            timeout = 5.0  # 5 seconds timeout
            start_time = time.time()
            
            while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
                if time.time() - start_time > timeout:
                    self.node.get_logger().error('Reset simulation service not available after timeout!')
                    raise RuntimeError("Reset simulation service not available")
                self.node.get_logger().warn('Reset simulation service not available, waiting...')
            
            while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
                if time.time() - start_time > timeout:
                    self.node.get_logger().error('Reset world service not available after timeout!')
                    raise RuntimeError("Reset world service not available")
                self.node.get_logger().warn('Reset world service not available, waiting...')
            
            # Pause physics first
            pause_future = self.pause_physics_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, pause_future, timeout_sec=1.0)
            
            # Reset simulation and world state
            reset_sim_future = self.reset_simulation_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, reset_sim_future, timeout_sec=1.0)
            
            reset_world_future = self.reset_world_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, reset_world_future, timeout_sec=1.0)
            
            # Wait for a moment to ensure reset is complete
            time.sleep(0.5)
            
            # Unpause physics
            unpause_future = self.unpause_physics_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, unpause_future, timeout_sec=1.0)
            
            # Additional spins to process any pending callbacks
            for _ in range(10):
                rclpy.spin_once(self.node, timeout_sec=0.1)
            
            return True
            
        except Exception as e:
            self.node.get_logger().error(f'Error during Gazebo reset: {str(e)}')
            return False

    def step(self, action):
        """Execute environment step"""
        self.steps_taken += 1
        
        # Apply action
        v, w = self.compute_dwa_velocities()[action]
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = w
        self.vel_pub.publish(twist_msg)

        # Allow ROS to process messages
        rclpy.spin_once(self.node, timeout_sec=self.dt)

        # Check for collision
        min_distance = min(self.laser_scan)
        is_collision = min_distance < self.collision_threshold

        # Get observation and compute reward
        observation = self.get_observation()
        reward = self.compute_reward(action)
        
        # Check terminal conditions
        done = False
        info = {}
        
        if is_collision:
            self.node.get_logger().warn('Collision detected! Resetting environment...')
            reward = self.reward_weights["collision"]
            done = True
            info['termination_reason'] = 'collision'
            
            # Reset Gazebo and environment
            if not self.reset_gazebo():
                self.node.get_logger().error('Failed to reset after collision!')
                info['reset_failed'] = True
            
        elif np.linalg.norm(self.goal - self.position) < self.goal_threshold:
            done = True
            info['termination_reason'] = 'goal_reached'
        elif self.steps_taken >= self.max_episode_steps:
            done = True
            info['termination_reason'] = 'max_steps_exceeded'

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.steps_taken = 0
        
        # Reset Gazebo simulation
        if not self.reset_gazebo():
            self.node.get_logger().error('Failed to reset Gazebo during environment reset!')
        
        # Clear history buffers
        self.velocity_history = []
        self.angular_velocity_history = []
        self.obstacle_history = []
        self.goal_alignment_history = []
        
        # Reset state
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.velocity = [0.0, 0.0]
        self.previous_distance = None
        
        # Set new random goal if specified
        if options and options.get('random_goal', False):
            self.goal = np.array([
                random.uniform(-4.0, 4.0),
                random.uniform(-4.0, 4.0)
            ])
        
        # Get initial observation and ensure ROS is updated
        observation = self.get_observation()
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        return observation, {}

    def shutdown(self):
        """Clean up resources"""
        # Stop the robot
        stop_msg = Twist()
        self.vel_pub.publish(stop_msg)
        
        # Destroy service clients
        self.reset_simulation_client.destroy()
        self.reset_world_client.destroy()
        self.pause_physics_client.destroy()
        self.unpause_physics_client.destroy()
        
        # Shutdown ROS
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # Example usage
    env = TurtleBot3DWAEnv()
    try:
        obs, _ = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            if done:
                print(f"Episode ended: {info['termination_reason']}")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        env.shutdown()