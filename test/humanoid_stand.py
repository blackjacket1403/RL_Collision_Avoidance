import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def train_agent():
    # Create the HumanoidStandup-v5 environment without rendering
    env = gym.make("HumanoidStandup-v5", render_mode=None)  # No rendering during training

    # Instantiate the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e6), progress_bar=True)

    # Save the trained agent
    model.save("ppo_humanoid_standup")
    del model  # Delete the trained model to demonstrate loading

    # Close the environment after training
    env.close()

def test_agent():
    # Create the HumanoidStandup-v5 environment with rendering enabled
    env = gym.make("HumanoidStandup-v5", render_mode="human")  # Enable rendering during testing

    # Load the trained agent
    model = PPO.load("ppo_humanoid_standup", env=env)

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