import gymnasium as gym
import warnings
import sys
import numpy as np
import os
import termios, tty
import jackal_env  # Ensure this imports the file that registers the env

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
# warnings.filterwarnings("ignore")

env = gym.make("JackalEnv-v0")
obs, _ = env.reset()
done = False

while not done:
    state = obs
    action = env.action_space.sample()
    obs, reward, done,_, info = env.step(action)
    # env.render()
env.close()
