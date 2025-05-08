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
sys.path.append(current_path+"/nlp_trajectory")

# Import the NLP interface
from improved_jackal_trajectory import JackalTrajectoryGenerator
from jackal_nlp_interface import get_trajectory, set_position, reset_position

# ---- Environment Setup ---- #
ENV_ID = "JackalEnv-v0"
ENABLE_RENDER = False

env = gym.make(ENV_ID)
model  = TD3.load("final_lr_100_wt_2.0_wc_0.0", env=env, print_system_info=True)

Nm = 500

success = 0
path_length = 0

for trial in range(Nm):
    print(f"Trial {trial}")
    done = False
    target = np.random.normal(0.0,3.0)
    reset_position()
    trajectory = get_trajectory(f"move forward {target} meter")
    obs, _ = env.reset(options={"waypoints": trajectory})

    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, _, info = env.step(action)
        state = env.unwrapped.getStandardState()
        # env.render()
    if(abs(state[1]-target)< 0.1):
        success += 1
        path_length += info["pos_error"]
print("Forward and backward Commands")
print(f"Success rate: {(success / Nm) * 100:.2f}%")
print(f"Average trakcing Error: {path_length / success:.2f} m")

print("Left turn Commands")
success = 0
tracking_error = 0
for trial in range(Nm):
    print(f"Trial {trial}")
    done = False
    reset_position()
    trajectory = get_trajectory(f"turn left 90 degrees")
    obs, _ = env.reset(options={"waypoints": trajectory})

    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, _, info = env.step(action)
        state = env.unwrapped.getStandardState()
        # env.render()
    if(abs(state[1]-target)< 0.1):
        success += 1
        path_length += info["pos_error"]

print(f"Success rate: {(success / Nm) * 100:.2f}%")
print(f"Average tracking Error: {path_length / success:.2f} m")
    
print("Right turn Commands")
success = 0
tracking_error = 0
for trial in range(Nm):
    print(f"Trial {trial}")
    done = False
    reset_position()
    trajectory = get_trajectory(f"turn right 90 degrees")
    obs, _ = env.reset(options={"waypoints": trajectory})

    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, _, info = env.step(action)
        state = env.unwrapped.getStandardState()
        # env.render()
    if(abs(state[1]-target)< 0.1):
        success += 1
        path_length += info["pos_error"]

print(f"Success rate: {(success / Nm) * 100:.2f}%")
print(f"Average tracking Error: {path_length / success:.2f} m")
    

print("Goto Commands")
success = 0
tracking_error = 0

for trial in range(Nm):
    print(f"Trial {trial}")
    done = False
    reset_position()
    trajectory = get_trajectory(f"turn right 90 degrees")
    obs, _ = env.reset(options={"waypoints": trajectory})

    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, _, info = env.step(action)
        state = env.unwrapped.getStandardState()
        # env.render()
    if(abs(state[1]-target)< 0.1):
        success += 1
        path_length += info["pos_error"]

print(f"Success rate: {(success / Nm) * 100:.2f}%")
print(f"Average tracking Error: {path_length / success:.2f} m")
    