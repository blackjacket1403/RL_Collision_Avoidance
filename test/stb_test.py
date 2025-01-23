import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def train_agent():
    # Create environment without rendering
    env = gym.make("BipedalWalker-v3", render_mode=None)  # No rendering during training

    # Instantiate the agent (using PPO instead of DQN)
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e6), progress_bar=True)

    # Save the agent
    model.save("ppo_bipedal_walker")
    del model  # delete trained model to demonstrate loading

    # Close the environment after training
    env.close()

def test_agent():
    # Create environment with render_mode set to "human"
    env = gym.make("BipedalWalker-v3", render_mode="human")  # Enable rendering during testing

    # Load the trained agent
    model = PPO.load("ppo_bipedal_walker", env=env)

    # Evaluate the agent (without rendering)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent with rendering in "human" mode after training
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()  # Render the environment during interaction

        if dones:
            obs = env.reset()

    # Close the environment after rendering
    env.close()

# Call these functions based on your needs
# Uncomment the function call to train the agent
#train_agent()

# Uncomment the function call to test the trained agent with rendering
test_agent()

