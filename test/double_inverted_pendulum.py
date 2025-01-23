import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

def train_agent():
    # Create the InvertedDoublePendulum-v5 environment without rendering
    env = gym.make("InvertedDoublePendulum-v5", render_mode=None)  # No rendering during training

    # Define action noise for DDPG (helps with exploration)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Instantiate the DDPG agent
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,  # Add action noise for exploration
        verbose=1,
    )

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)

    # Save the trained agent
    model.save("ddpg_double_pendulum")
    del model  # Delete the trained model to demonstrate loading

    # Close the environment after training
    env.close()

def test_agent():
    # Create the InvertedDoublePendulum-v5 environment with rendering enabled
    env = gym.make("InvertedDoublePendulum-v5", render_mode="human")  # Enable rendering during testing

    # Load the trained agent
    model = DDPG.load("ddpg_double_pendulum", env=env)

    # Evaluate the agent (without rendering)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # Enjoy the trained agent with rendering in "human" mode
    obs, _ = env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()  # Render the environment during interaction

        if dones or truncated:
            obs, _ = env.reset()

    # Close the environment after rendering
    env.close()

# Call these functions based on your needs
# Uncomment the function call to train the agent
#train_agent()

# Uncomment the function call to test the trained agent with rendering
test_agent()