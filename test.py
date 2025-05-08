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
import time
from typing import List, Tuple

# Ensure path includes the current directory
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

# Import the NLP interface
from nlp_trajectory.improved_jackal_trajectory import JackalTrajectoryGenerator
from nlp_trajectory.jackal_nlp_interface import get_trajectory, set_position, reset_position

# ---- Environment Setup ---- #
ENV_ID = "JackalEnv-v0"
ENABLE_RENDER = False

env = gym.make(ENV_ID)
model  = TD3.load("final_lr_100_wt_2.0_wc_0.0", env=env, print_system_info=True)

done = False
target = np.random.normal(0.0,3.0)
trajectory = get_trajectory(f"move forward {target} meters")
obs, _ = env.reset(options={"Waypoints": trajectory})
# obs, _ = env.reset()

while not done:
    action = model.predict(obs, deterministic=True)[0]
    # action = np.array([0.0, 0.0])

    obs, reward, done, _, info = env.step(action)

    env.render()

    