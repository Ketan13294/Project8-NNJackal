import gymnasium as gym
import warnings
import sys
import os
import jackal_env  # Ensure this imports the file that registers the env

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
# warnings.filterwarnings("ignore")

env = gym.make("JackalEnv-v0")
obs, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done,_, _ = env.step(action)
    print("State: [",obs[0],obs[1],obs[2],"]")
    #env.render()
