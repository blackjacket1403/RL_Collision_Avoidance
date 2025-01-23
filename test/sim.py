import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# PPO Network
class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Outputs actions in the range [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value


# Environment Setup
class Environment:
    def __init__(self, width, height, num_obstacles):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.reset_environment()

    def reset_environment(self):
        # Randomize start, goal, and obstacles
        self.start = (random.uniform(0, self.width), random.uniform(0, self.height))
        self.goal = (random.uniform(0, self.width), random.uniform(0, self.height))
        self.obstacles = self.generate_obstacles(self.num_obstacles)
        self.agent = self.Agent(self.start[0], self.start[1], self.width, self.height)

    def generate_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            obstacles.append((x, y))
        return obstacles

    class Agent:
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.vx = 0
            self.vy = 0
            self.radius = 5
            self.lidar_range = 50
            self.lidar_angles = np.linspace(0, 2 * np.pi, 36)  # 36 LiDAR beams
            self.width = width
            self.height = height

        def get_lidar_readings(self, obstacles):
            readings = []
            for angle in self.lidar_angles:
                dx = np.cos(angle)
                dy = np.sin(angle)
                dist = 0
                while dist < self.lidar_range:
                    dist += 1
                    x = self.x + dx * dist
                    y = self.y + dy * dist
                    if any(np.sqrt((x - ox) ** 2 + (y - oy) ** 2) < 10 for ox, oy in obstacles):
                        break
                readings.append(dist)
            return readings

        def move(self, vx, vy):
            # Update velocity
            self.vx = vx
            self.vy = vy

            # Update position based on velocity
            self.x += self.vx
            self.y += self.vy

            # Boundary check and reset velocity if out of bounds
            if self.x <= 0 or self.x >= self.width:
                self.vx = 0  # Stop horizontal movement
                self.x = np.clip(self.x, 0, self.width)  # Clip position to boundary

            if self.y <= 0 or self.y >= self.height:
                self.vy = 0  # Stop vertical movement
                self.y = np.clip(self.y, 0, self.height)  # Clip position to boundary

        def reset(self):
            self.x = self.width / 2
            self.y = self.height / 2
            self.vx = 0
            self.vy = 0


# DWA-PPO Algorithm
class DWA_PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, clip_epsilon):
        self.ppo = PPO(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.ppo.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.ppo.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = deque(maxlen=10000)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, _ = self.ppo(state)
        action = action_mean.detach().numpy()[0]
        return action

    def update(self):
        if len(self.memory) < 64:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        _, values = self.ppo(states)
        _, next_values = self.ppo(next_states)

        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values

        action_means, _ = self.ppo(states)
        dist = torch.distributions.Normal(action_means, 1.0)
        log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
        old_log_probs = dist.log_prob(actions.detach()).sum(dim=1, keepdim=True)
        ratios = torch.exp(log_probs - old_log_probs)

        surr1 = ratios * advantages.unsqueeze(1)
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.unsqueeze(1)
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = nn.MSELoss()(values.squeeze(), returns.squeeze())

        self.optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)  # Retain graph for multiple backward passes
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.memory.clear()


# Simulation and Visualization
def simulate(env, dwa_ppo, num_episodes):
    fig, ax = plt.subplots()
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    agent_circle = patches.Circle((env.agent.x, env.agent.y), env.agent.radius, color='blue')
    goal_circle = patches.Circle(env.goal, 10, color='green')
    ax.add_patch(agent_circle)
    ax.add_patch(goal_circle)
    obstacles = [patches.Circle(obs, 10, color='red') for obs in env.obstacles]
    for obs in obstacles:
        ax.add_patch(obs)

    def update(frame):
        state = env.agent.get_lidar_readings(env.obstacles)
        action = dwa_ppo.select_action(state)
        env.agent.move(action[0], action[1])

        next_state = env.agent.get_lidar_readings(env.obstacles)

        # Reward components
        distance_to_goal = np.sqrt((env.agent.x - env.goal[0]) ** 2 + (env.agent.y - env.goal[1]) ** 2)
        reward_distance = -distance_to_goal  # Encourage moving toward the goal

        # Penalize for being close to obstacles
        min_obstacle_distance = min(env.agent.get_lidar_readings(env.obstacles))
        reward_obstacle = -10 / (min_obstacle_distance + 1e-5)  # Penalize proximity to obstacles

        # Penalize for being near boundaries
        boundary_penalty = 0
        if env.agent.x <= 10 or env.agent.x >= env.width - 10 or env.agent.y <= 10 or env.agent.y >= env.height - 10:
            boundary_penalty = -10  # Penalize being near boundaries

        # Total reward
        reward = reward_distance + reward_obstacle + boundary_penalty

        # Check if goal is reached
        done = distance_to_goal < 10

        dwa_ppo.memory.append((state, action, reward, next_state, done))
        dwa_ppo.update()

        agent_circle.center = (env.agent.x, env.agent.y)
        if done:
            print("Goal reached! Resetting environment.")
            env.reset_environment()
            agent_circle.center = (env.agent.x, env.agent.y)
            goal_circle.center = env.goal
            for obs, patch in zip(env.obstacles, obstacles):
                patch.center = obs

        # Print agent position and reward
        print(f"Agent Position: ({env.agent.x:.2f}, {env.agent.y:.2f}), Reward: {reward:.2f}")

        return agent_circle, goal_circle, *obstacles

    ani = FuncAnimation(fig, update, frames=num_episodes, blit=True, repeat=False)
    plt.show()


# Main Execution
if __name__ == "__main__":
    env = Environment(200, 200, 5)
    dwa_ppo = DWA_PPO(state_dim=36, action_dim=2, lr_actor=0.001, lr_critic=0.001, gamma=0.99, clip_epsilon=0.2)
    simulate(env, dwa_ppo, num_episodes=1000)