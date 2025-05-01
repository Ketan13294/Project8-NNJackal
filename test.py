import gymnasium as gym
import warnings
import sys
import numpy as np
import os
import jackal_env  # Ensure this imports the file that registers the env

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
# warnings.filterwarnings("ignore")

env = gym.make("JackalEnv-v0")
obs, _ = env.reset()
print(obs)
done = False

from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D


def generate_random_waypoints(duration=30.0, dt=3.0, max_speed=2.0, num_waypoints=10):

    t_points = np.linspace(0, duration, num_waypoints)
    x_points = np.cumsum(np.random.uniform(0.5, max_speed, size=num_waypoints))
    y_points = np.cumsum(np.random.uniform(-0.2, 0.2, size=num_waypoints))  # Reduced range for smoother transitions
    
    spline_x = CubicSpline(t_points, x_points, bc_type='natural')  # Natural boundary conditions for smoother curvature
    spline_y = CubicSpline(t_points, y_points, bc_type='natural')

    t_query = np.arange(0, duration, 0.001)
    x = spline_x(t_query)
    y = spline_y(t_query)

    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)

    waypoints = list(zip(t_query, x, y, yaw))
    return waypoints


waypoints = generate_random_waypoints(duration=30.0, dt=3.0, max_speed=2.0, num_waypoints=10)
import matplotlib.pyplot as plt

# Extract time, x, and y coordinates from waypoints
time_coords = [0*wp[0] for wp in waypoints]
x_coords = [wp[1] for wp in waypoints]
y_coords = [wp[2] for wp in waypoints]

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the waypoints in 3D
ax.plot(x_coords, y_coords, time_coords, marker='o', label='Waypoints Path')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Time')
ax.set_title('3D Waypoints Path')
ax.legend()
plt.show()

# env.unwrapped.setReferenceWaypoints([[3.0,5.0,0.0,0.0],[10.0,10.0,0.0,0.0]])

# while not done:
#     # action = env.action_space.sample()
#     state = obs
#     action = [0.1*(10.0-state[1]),0.1*(10.0-state[1])]
#     obs, reward, done,_, info = env.step(action)
#     # print("Time: ",state[0]," State: [",obs[1],obs[2],"], Reward: ",reward," Time: ",info["time_error"]," Pos: ",info["pos_error"]," Angle: ",info["angle_error"]," Control: ",info["control_penalty"])
#     env.render()

# env.close()
