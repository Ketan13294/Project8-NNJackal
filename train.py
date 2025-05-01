import gymnasium as gym
import jackal_env
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt

# ---- Environment Setup ---- #
ENV_ID = "JackalEnv-v0"
ENABLE_RENDER = True

def generate_random_waypoints(duration=30.0, dt=3.0, max_speed=2.0, num_waypoints=10):

    t_points = np.linspace(0, duration, num_waypoints)
    x_points = np.cumsum(np.random.uniform(0.5, max_speed, size=num_waypoints))
    y_points = np.cumsum(np.random.uniform(-0.2, 0.2, size=num_waypoints))  # Reduced range for smoother transitions
    
    spline_x = CubicSpline(t_points, x_points, bc_type='natural')  # Natural boundary conditions for smoother curvature
    spline_y = CubicSpline(t_points, y_points, bc_type='natural')

    t_query = np.arange(0, duration, dt)
    x = spline_x(t_query)
    y = spline_y(t_query)

    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)

    waypoints = list(zip(t_query, x, y, yaw))
    return waypoints


# ---- Callback to print reward ---- #
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, env, verbose=1, plot_path="td3_jackal_reward_plot.png",save_freq=10):
        super().__init__(verbose)
        self.env = env
        self.verbose = verbose
        self.plot_path = plot_path
        self.episode_rewards = []
        self.episode_reward = 0.0
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]

        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % self.save_freq == 0:
            self.model.save(f"checkpoints/td3_jackal_trajectory_{len(self.episode_rewards)}")
            if self.verbose > 0:
                print(f"Model saved at episode {len(self.episode_rewards)}")

        if ENABLE_RENDER:
            self.env.render()  # Use your MuJoCo renderer here

        if self.locals["dones"][0]:
            print(f"\033[92m[Episode {len(self.episode_rewards) + 1} reward: {self.episode_reward:.2f}]\033[0m")
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0

            # Set new random trajectory for next episode
            trajectory = generate_random_waypoints()
            self.env.unwrapped.setReferenceWaypoints(trajectory)

        return True

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, label='Episode reward')
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("TD3 Training Rewards")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()

env = gym.make(ENV_ID)

# Initial trajectory
trajectory = generate_random_waypoints()
env.unwrapped.setReferenceWaypoints(trajectory)

# Add noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions))

# TD3 Model
model = TD3(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    buffer_size=100_000,
    learning_rate=1e-3,
    batch_size=100,
    train_freq=(1,"episode"),
    tau=0.01,
    gradient_steps=-1,
    gamma=0.99,
    policy_delay=10,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    seed = 42,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

# Train with reward printing and trajectory update
callback = EpisodeRewardCallback(env=env,verbose=1)
model.learn(total_timesteps=500_000_000, callback=callback)

# Save model
model.save("td3_jackal_trajectory")
