import gymnasium as gym
import jackal_env
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from matplotlib import pyplot as plt
import random
import os
import datetime

# ---- Environment Setup ---- #
ENV_ID = "JackalEnv-v0"
ENABLE_RENDER = False

# ---- Callback to log reward ---- #
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, env, verbose=True, plot_path="td3_jackal_reward_plot"):
        super().__init__(verbose)
        self.env = env
        self.episode_rewards = []
        self.episode_reward = 0.0
        self.plot_path = plot_path

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]

        if ENABLE_RENDER:
            self.env.render()

        if self.locals["dones"][0]:
            print(f"\033[92m[Episode {len(self.episode_rewards) +1} reward: {self.episode_reward:.2f}]\033[0m")
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0

            # Plot rewards
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, label='Episode reward')
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("TD3 Training Rewards")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.plot_path}.png")
            plt.close()
        return True

env = gym.make(ENV_ID)

# Add noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Generate timestamp for unique checkpoint folders
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# TD3 Model with LARGE network
model = TD3(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=dict(net_arch=dict(
        pi=[256, 256, 128, 128, 64, 64],
        qf=[1024, 512, 256, 128]
    )),
    action_noise=action_noise,
    verbose=1,
    buffer_size=1_000_000,
    learning_rate=4e-5,
    batch_size=4096,
    train_freq=(4, "step"),
    tau=0.002,
    gamma=0.99,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    seed=42,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

# ---- Checkpoint Callback ---- #
checkpoint_path = f'./checkpoints/checkpoints_{timestamp}/'
os.makedirs(checkpoint_path, exist_ok=True)

model_path = f'./models/best_model_{timestamp}/'
os.makedirs(model_path, exist_ok=True)

logs_path = f'./logs/log_{timestamp}/'
os.makedirs(logs_path, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=12_000,  # save every 10,000 steps
    save_path=checkpoint_path,
    name_prefix=f'td3_jackal_{timestamp}'
)

# ---- Eval Callback ---- #
eval_callback = EvalCallback(
    env,
    best_model_save_path=model_path,
    log_path=logs_path,
    eval_freq=60_000,  # evaluate every 50,000 steps
    deterministic=True,
    render=False
)

# ---- Training loop with all callbacks ---- #
reward_callback = EpisodeRewardCallback(env=env, verbose=True)

callback = CallbackList([reward_callback, checkpoint_callback, eval_callback])

model.learn(
    total_timesteps=3_000_000,
    callback=[reward_callback, checkpoint_callback, eval_callback]
)

# Save final model
model.save(f"td3_jackal_trajectory_large_{timestamp}")
