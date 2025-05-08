import numpy as np
import matplotlib.pyplot as plt
import math

class TrajectoryGenerator:
    def generate_random_waypoints(self, state=np.zeros((7,1)), duration=5.5, dt=0.01, max_lin_vel=1.0, max_ang_vel=0.5):
        n_points = int(duration / dt)+1
        t_query = np.linspace(state[0], state[0]+duration, n_points)

        lin_vel = np.clip(np.random.normal(0.0,0.5)*np.ones(n_points),-max_lin_vel,max_lin_vel)
        ang_vel = 0*np.clip(np.random.normal(0.0,0.2, n_points), -max_ang_vel, max_ang_vel)

        x = [state[1]]
        y = [state[2]]
        yaw = [state[3]]
        for i in range(1, n_points):
            dt_i = t_query[i] - t_query[i - 1]
            theta = yaw[-1]
            x.append(x[-1] + lin_vel[i] * np.cos(theta) * dt_i)
            y.append(y[-1] + lin_vel[i] * np.sin(theta) * dt_i)
            yaw.append(yaw[-1] + ang_vel[i] * dt_i)
        
        yaw = np.array(yaw)
        yaw_ = ((yaw + math.pi) % (2 * math.pi)) - math.pi
        yaw = yaw_.tolist()

        waypoints = np.array([list(item) for item in zip(t_query, x, y, yaw)])
        print(waypoints[:,1:3])
        return waypoints

# === TEST SCRIPT ===
generator = TrajectoryGenerator()

# Initial state: t=0, x=0, y=0, theta=0
initial_state = np.zeros((7,1))
waypoints = np.array(generator.generate_random_waypoints(state=initial_state))

# Extract arrays
t = waypoints[:, 0]
x = waypoints[:, 1]
y = waypoints[:, 2]
theta = waypoints[:, 3]

# Compute arrow components for heading
arrow_len = 0.1
u = np.cos(theta) * arrow_len
v = np.sin(theta) * arrow_len

# === Plot ===
plt.figure(figsize=(10,10))
plt.plot(x, y, color='gray', alpha=0.6, label='Trajectory')
plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='blue', width=0.005)

plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Generated Random Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
