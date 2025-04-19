import gymnasium as gym
import sys
import os
import jackal_env  # Ensure this imports the file that registers the env

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

env = gym.make("JackalEnv-v0")
obs, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    env.render()