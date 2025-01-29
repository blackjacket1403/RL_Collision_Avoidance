import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO  # Assuming you used PPO for training
from turtlebot_env import TurtleBot3DWAEnv # Import your custom environment

# Path to your trained model
MODEL_PATH = "turtlebot3_sac"  # Replace with the correct path if needed

def test_model(env, model, num_episodes=10):
    """Test the trained model in the environment."""
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)  # Use deterministic actions for testing
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Print step information (optional)
            print(f"Step {step_count}: Action={action}, Reward={reward}, Done={done}, Truncated={truncated}")

            if done:
                print(f"Episode ended early! Reason: Done={done}, Truncated={truncated}")

        print(f"Episode {episode + 1} finished. Total Reward: {total_reward}, Steps: {step_count}\n")

    print("Testing completed.")

if __name__ == "__main__":
    # Create the environment
    env = TurtleBot3DWAEnv()

    # Load the trained model
    model = PPO.load(MODEL_PATH, env=env)  # Load the PPO model

    # Test the model
    test_model(env, model, num_episodes=5)  # Test for 5 episodes

    # Close the environment
    env.close()