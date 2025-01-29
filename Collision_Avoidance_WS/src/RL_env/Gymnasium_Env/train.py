from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from turtlebot_env import TurtleBot3DWAEnv
import matplotlib.pyplot as plt
import numpy as np
import os

# Create log directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Create the environment
env = TurtleBot3DWAEnv()

# Wrap the environment with Monitor to log rewards
env = Monitor(env, log_dir)

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, device='cuda:0')

# Train the agent
print("Training the agent...")
model.learn(total_timesteps=200000)

# Save the trained model
model.save("models/turtlebot3_sac")
print("Model saved to models/turtlebot3_sac")

# Close the environment
env.close()

# Load the rewards from the Monitor logs
def load_rewards(log_dir):
    """
    Load rewards from the Monitor logs.
    """
    rewards = []
    with open(os.path.join(log_dir, "monitor.csv"), "r") as f:
        lines = f.readlines()
        for line in lines[2:]:  # Skip the header lines
            episode_reward = float(line.split(",")[0])
            rewards.append(episode_reward)
    return rewards

# Plot the rewards vs episodes
def plot_rewards(rewards, window_size=100):
    """
    Plot the rewards vs episodes with a moving average.
    """
    # Calculate moving average
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    # Plot raw rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Reward", alpha=0.3, color="blue")

    # Plot moving average
    plt.plot(moving_avg, label=f"Moving Average (window={window_size})", color="red")

    # Add labels and title
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episodes")
    plt.legend()
    plt.grid()
    plt.show()

# Load rewards and plot
rewards = load_rewards(log_dir)
plot_rewards(rewards)