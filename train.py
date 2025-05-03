import gymnasium as gym
import jackal_env
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt

# ---- Environment Setup ---- #
ENV_ID = "JackalEnv-v0"
ENABLE_RENDER = False


# ---- Callback to print reward ---- #
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, env, verbose=True, plot_path="td3_jackal_reward_plot",save_freq=50):
        super().__init__(verbose)
        self.env = env
        self.verbose = verbose
        self.plot_path = plot_path
        self.episode_rewards = []
        self.episode_reward = 0.0
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]

        # if len(self.episode_rewards) > 0 and len(self.episode_rewards) % self.save_freq == 0:
        #     self.model.save(f"checkpoints/td3_jackal_trajectory_{len(self.episode_rewards)}")
        #     if self.verbose > 0:
        #         print(f"Model saved at episode {len(self.episode_rewards)}")

        if ENABLE_RENDER:
            self.env.render()  # Use your MuJoCo renderer here

        if self.locals["dones"][0]:
            print(f"\033[92m[Episode {len(self.episode_rewards) + 1} reward: {self.episode_reward:.2f}]\033[0m")
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0
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
                print(f"Reward plot saved to {self.plot_path+"_{len(self.episode_rewards)}.png"}")
        return True

env = gym.make(ENV_ID)

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
    train_freq=(200, "step"),
    tau=0.01,
    gamma=0.99,
    policy_delay=10,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    seed = 42,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

n_episodes = 40000
n_max_steps = 5000

CheckpointCallback = CheckpointCallback(save_freq=1000, save_path="checkpoints/", name_prefix="td3_traj_follow")

EvalCallback = EvalCallback(env, best_model_save_path="best_model/",
                              log_path="logs/", eval_freq=500)

callback = EpisodeRewardCallback(env=env,verbose=True)

for episode in range(n_episodes):
    model.learn(total_timesteps=n_max_steps, callback=callback,progress_bar=False, log_interval=100)

# Save model
model.save("td3_jackal_trajectory")
