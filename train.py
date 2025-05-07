import gymnasium as gym
import jackal_env
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList,StopTrainingOnNoModelImprovement
from matplotlib import pyplot as plt
import random
import os
import datetime
import sys

# ---- Environment Setup ---- #
ENV_ID = "JackalEnv-v0"
ENABLE_RENDER = False

# ---- Callback to log reward ---- #
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, env, verbose=True, plot_path="./plots/td3_jackal_reward_plot"):
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

            plt.figure(figsize=(10, 5))
            # Calculate moving average
            window_size = 50
            if len(self.episode_rewards) >= window_size:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window_size) / window_size, mode='valid')
                plt.plot(range(window_size - 1, len(self.episode_rewards)), moving_avg, label='Moving Average Reward', color='darkblue')

            # Plot raw rewards
            plt.plot(self.episode_rewards, label='Raw Episode Reward', color='lightblue', alpha=0.6)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("TD3 Training Rewards")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.plot_path}.png")
            plt.close()
        return True

if len(sys.argv) < 5:
    print("Usage: python train.py <wt> <wc> <wv> <wl> <Lr>")
    sys.exit(1)

wt = sys.argv[1]
wc = sys.argv[2]
wv = sys.argv[3]
wl = sys.argv[4]
Lr = float(sys.argv[5])

# Generate timestamp for unique checkpoint folders
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


if (len(sys.argv) == 7):
    additional = str(sys.argv[6])
    print(additional)
    folder_name=f"{timestamp}_lr_{int(float(Lr)*10**6)}_wt_{wt}_wc_{wc}_wv_{wv}_wl_{wl}_{additional}"
else:
    folder_name=f"{timestamp}_lr_{int(float(Lr)*10**6)}_wt_{wt}_wc_{wc}_wv_{wv}_wl_{wl}"


print(f"Wt: {wt}")
print(f"Wc: {wc}")
print(f"Wv: {wv}")
print(f"Wl: {wl}")
print(f"Lr: {Lr}")
print(f"Folder_name: {folder_name}")
load_model_path = "training_data/models/lr_1_wt_2.0_wc_1.0/20250505_011406/best_model"
tau = 0.002
env = gym.make(ENV_ID, wt=float(wt), wc=float(wc),wv=float(wv),wl=float(wl))

# Add noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

# TD3 Model with LARGE network
#model = TD3(
#    policy="MlpPolicy",
#    env=env,
#    policy_kwargs=dict(net_arch=dict(
#        pi=[256, 256, 128, 128, 64, 64],
#        qf=[1024, 512, 256, 128]
#    )),
#    action_noise=action_noise,
#    verbose=1,
#    buffer_size=1_000_000,
#    learning_rate=Lr,
#    batch_size=4096,
#    train_freq=(4, "step"),
#    tau=0.0001,
#    gamma=0.99,
#    policy_delay=2,
#    target_policy_noise=0.2,
#    target_noise_clip=0.5,
#    seed=random.randint(0, 10000),
#    device="cuda:0" if torch.cuda.is_available() else "cpu",
#)
model = TD3.load("training_data/models/lr_100_wt_2.0_wc_0.1_TimeExpLateralWvMediumConditional/20250507_013939/best_model",
                 env=env,
                 print_system_info=True,
                 learning_rate=Lr,
                 tau=tau)

# ---- Directories ---- #
checkpoint_path = f"training_data/checkpoints/{folder_name}"
os.makedirs(checkpoint_path, exist_ok=True)

model_path = f"training_data/models/{folder_name}"
os.makedirs(model_path, exist_ok=True)
os.makedirs(model_path+f"/{timestamp}",exist_ok=True)

logs_path = f"training_data/logs/{folder_name}"
os.makedirs(logs_path, exist_ok=True)

plot_path = f"training_data/plots/{timestamp}_{folder_name}"
os.makedirs(plot_path, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=6000,  # save every 6000 steps
    save_path=checkpoint_path,
    name_prefix=f"td3_jackal_{timestamp}"
)


# ---- Stop Training Callback ---- #
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=100, verbose=1)

# ---- Eval Callback ---- #
eval_callback = EvalCallback(
    env,
    best_model_save_path=model_path+f"/{timestamp}",
    log_path=logs_path,
    eval_freq=6000,  # evaluate every 6000 steps
    deterministic=True,
    render=False,
    callback_after_eval=stop_train_callback
)

# ---- Training loop with all callbacks ---- #
reward_callback = EpisodeRewardCallback(env=env, verbose=True, plot_path=plot_path+f"/{timestamp}_training_plot_{folder_name}")


callback = CallbackList([reward_callback, checkpoint_callback, eval_callback])

model.learn(
    total_timesteps=2_000_000,
    callback=callback
)

# Save final model
model.save(model_path+f"/{timestamp}/{timestamp}_final_{folder_name}")