import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import jackal_env
from scipy.interpolate import CubicSpline
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# ---- Environment Setup ---- #
ENV_ID = "JackalEnv-v0"
ENABLE_RENDER = False

def generate_random_waypoints(duration=10.0, dt=0.1, max_speed=1.0, num_waypoints=10):

    t_points = np.linspace(0, duration, num_waypoints)
    x_points = np.cumsum(np.random.uniform(0.5, max_speed, size=num_waypoints))
    y_points = np.random.uniform(-2.0, 2.0, size=num_waypoints)
    
    spline_x = CubicSpline(t_points, x_points)
    spline_y = CubicSpline(t_points, y_points)

    t_query = np.arange(0, duration, dt)
    x = spline_x(t_query)
    y = spline_y(t_query)

    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)

    waypoints = list(zip(x, y, yaw))
    return waypoints


# ---- Callback to print reward ---- #
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, env,plot_path="td3_jackal_reward_plot.png", verbose=1):
        super().__init__(verbose)
        self.env = env
        self.plot_path = plot_path
        self.episode_rewards = []
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]

        if ENABLE_RENDER:
            self.env.render()  # Use your MuJoCo renderer here

        if self.locals["dones"][0]:
            print(f"\033[92m[Episode {len(self.episode_rewards) + 1} reward: {self.episode_reward:.2f}]\033[0m")
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0

            # Set new random trajectory for next episode
            trajectory = generate_random_waypoints()
            self.env.envs[0].unwrapped.setReferenceWaypoints(trajectory)

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

env = make_vec_env(ENV_ID,n_envs=1)

# Initial trajectory
trajectory = generate_random_waypoints()
env.envs[0].unwrapped.setReferenceWaypoints(trajectory)

# Add noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# TD3 Model
model = TD3(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    buffer_size=100_000,
    learning_rate=1e-3,
    batch_size=100,
    tau=0.005,
    gamma=0.99,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
)

# Train with reward printing and trajectory update
callback = EpisodeRewardCallback(env=env,verbose=1)
model.learn(total_timesteps=500_000, callback=callback)

# Save model
model.save("td3_jackal_trajectory")
