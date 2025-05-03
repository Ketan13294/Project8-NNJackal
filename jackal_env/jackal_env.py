from mujoco import MjModel, MjData 
from mujoco import viewer
import mujoco

from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from copy import deepcopy
from gymnasium import spaces
import numpy as np
import time
import gymnasium
import io
import os

def theta2vec(theta):
    ''' Convert an angle (in radians) to a unit vector in that angle around Z '''
    return np.array([np.cos(theta), np.sin(theta), 0.0])


class JackalEnv(gymnasium.Env):
    def __init__(self):
        abs_path = os.path.dirname(__file__)

        self.model = MjModel.from_xml_path(f'{abs_path}/jackal.xml')
        self.model.opt.timestep = 0.001
        self.data = MjData(self.model)
        self.sim_time = 0.0

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        self.time_step = 0.01
        self.n_substeps = 10
        self.viewer = None

        # for environment
        self.control_freq = 100
        self.num_time_step = int(1.0/(self.time_step*self.control_freq))
        self.max_steps = 2000
        self.cur_step = 0
        self.path_length = 0.0

        # for state
        self.robot_pose = np.zeros(3)
        self.robot_vel = np.zeros(3)

        # Trajectory tracking state encoding
        self.horizon = 10.0
        self.max_lateral_error = 0.5
        self.encoding_dim = self.horizon + len(self.robot_vel)

        # for action
        self.action = np.zeros(2)

        # state & action dimension
        self.action_dim = 2
        #Waypoints(4 each), robot velocities (linear and angular), robot properties (length, width, wheel diameter)
        # 4*10 + 2 + 3 = 45
        self.state_dim = 4*self.horizon+2+3
        self.action_space = spaces.Box(-10*np.ones(self.action_dim), 10*np.ones(self.action_dim), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.state_dim), np.inf*np.ones(self.state_dim), dtype=np.float64)

        # reference points list
        self.reference_waypoints = self.generate_random_waypoints(state=self.robot_pose)


    def reset(self,seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.action = np.zeros(2)
        self.robot_vel = np.zeros(3)
        self.robot_pose = np.zeros(3)
        self.cur_step = 0
        self.path_length = 0.0
        self.sim_time = 0.0
        self.reference_waypoints.clear()
        state = self.getFlattenState()
        self.reference_waypoints = self.generate_random_waypoints(state)
        return (state,{})

    def render(self, **kwargs):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)

        # clear existing markers
        self.viewer.sync()

    def generate_random_waypoints(state=[0,0,0] ,duration=20, dt=0.01, max_lin_vel=1.0, max_ang_vel=0.5):
        n_points = int(duration / dt)
        t_query = np.linspace(0, duration, n_points)

        lin_vel = np.clip(np.random.normal(0.6, 0.2, n_points), 0, max_lin_vel)
        ang_vel = np.clip(np.random.normal(0.0, 0.2, n_points), -max_ang_vel, max_ang_vel)

        x, y, yaw = state
        for i in range(1, n_points):
            dt_i = t_query[i] - t_query[i - 1]
            theta = yaw[-1]
            x.append(x[-1] + lin_vel[i] * np.cos(theta) * dt_i)
            y.append(y[-1] + lin_vel[i] * np.sin(theta) * dt_i)
            yaw.append(yaw[-1] + ang_vel[i] * dt_i)

        waypoints = list(zip(t_query, x, y, yaw))
        return waypoints

    def getSensor(self):
        sensor_dict = {'accelerometer':None, 'velocimeter':None, 'gyro':None}
        for sensor_name in sensor_dict.keys():
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,sensor_name)
            adr = self.model.sensor_adr[id]
            dim = self.model.sensor_dim[id]
            sensor_dict[sensor_name] = self.data.sensordata[adr:adr + dim].copy()
        return sensor_dict

    def getStandardState(self):
        mujoco.mj_forward(self.model, self.data)
        sensor_dict = self.getSensor()
        self.robot_vel[0] = sensor_dict['velocimeter'][0]
        self.robot_vel[1] = sensor_dict['velocimeter'][1]
        self.robot_vel[2] = sensor_dict['gyro'][2]

        self.robot_pose = self.data.xpos[self.body_id].copy()
        robot_mat = self.data.xmat[self.body_id].copy().reshape(3, 3)
        theta = Rotation.from_matrix(robot_mat).as_euler('zyx', degrees=False)[0]
        self.robot_pose[2] = theta

        pos = deepcopy(self.robot_pose)
        vel = deepcopy(self.robot_vel)
        state = np.concatenate([self.sim_time],pos,vel)
        return state

    def getState(self):
        state = self.getStandardState()
        z_t = self.get_reference_point(state)
        # Robot Length, Robot Width, Wheel Diameter
        o_d = [0.508,0.43,0.1]
        e_state = np.concatenate([z_t, np.linalg.norm(state[4:6]),state[6],o_d], axis=0,dtype=np.float64)
        return e_state

    def get_reference_point(self, state):
        """
            Get the horizon length trajectory segment from the current time step
            Args:
            state (np.array): Current robot position [t, x, y, theta]

            Returns:
            ref_points: Segment of the trajectory of horizon STEPS
        """        
        future_points = [p for p in self.reference_waypoints if p[0] >= state[0]]
        
        # If no future points are found, use the current position of the robot but with zero orientation
        if len(future_points) == 0:
            future_points = np.tile([state[0],state[1],state[2],state[3]],(self.horizon,1))
            for i in range(self.horizon):
                future_points[i][0] = state[0] + self.time_step*self.num_time_step*(i+1) 
            ref_point = future_points

        # If there are future points but not enough to fill the horizon, fill the rest with the last point
        elif len(future_points)  < self.horizon:
            ref_point = future_points
            dt = ref_point[-1][0] - ref_point[-2][0]
            for _ in range(self.horizon - len(ref_point)):
                ref_point.append(ref_point[-1].copy())
                ref_point[-1][0] += dt
        
        # If there are enough future points, select horizon points
        else:
            time_diffs = [abs(p[0] - state[0]) for p in future_points]
            idx = int(np.argmin(time_diffs))
            ref_point = self.reference_waypoints[idx:idx + self.horizon]

        R = Rotation.from_euler('z', state[3], degrees=False).as_matrix()[:2,:2]
        Rt = -R@(state[1:3])
        ref_point_lin = np.array([[],[],[],[]])
        for i in  range(len(ref_point)):
            ref_point[i][0] = ref_point[i][0] - state[0]
            ref_point[i][1:3] = (R@(ref_point[i][1:3]) + Rt)
            ref_point[i][3] = ref_point[i][3] - state[3]
            ref_point[0].append(ref_point[i][0])
            ref_point[1].append(ref_point[i][1])
            ref_point[2].append(ref_point[i][2])
            ref_point[3].append(ref_point[i][3])

        ref_point = np.array(ref_point).flatten()
        return ref_point


    def compute_tracking_error(self,state):
        tree = KDTree([p[0] for p in self.reference_waypoints if p[0] >= state[0]])
        _, idx = tree.query(state[0])
        nearest_pt = self.reference_waypoints[idx]
        return np.linalg.norm(state[1:3] - nearest_pt[1:3])
    
    def compute_orientation_error(self,state):
        tree = KDTree([p[0] for p in self.reference_waypoints if p[0] >= state[0]])
        _, idx = tree.query(state[0])
        nearest_pt = self.reference_waypoints[idx]
        return abs(state[3] - nearest_pt[3])

    def step(self, action):
        self.cur_step += 1
        self.sim_time += self.time_step * self.num_time_step

        prev_state = self.data.xpos[self.body_id]
        self.path_length = 0.0
        weight = 0.7
        for j in range(self.n_substeps):
            mujoco.mj_forward(self.model, self.data)

            self.action[0] = weight*self.action[0] + (1.0 - weight)*np.clip(action[0], -1.5, 1.5)
            self.action[1] = weight*self.action[1] + (1.0 - weight)*np.clip(action[1], -1.5, 1.5)

            self.data.ctrl[0] = self.action[0]
            self.data.ctrl[1] = self.action[1]
            mujoco.mj_step(self.model,self.data)
            self.path_length += np.linalg.norm(self.data.xpos[self.body_id] - prev_state)
            prev_state = self.data.xpos[self.body_id].copy()

	    # Get current state
        state = self.getStandardState()
        v_left =  (self.action[0] + 1.5)/(1.5-(-1.5))
        v_right = (self.action[1] + 1.5)/(1.5-(-1.5))

	    # Check termination and done conditions
        goal_met = (np.linalg.norm(state[1:4] - self.reference_waypoints[99][1:4]) < 0.01)

        done = goal_met or (self.cur_step >= self.max_steps)
        is_terminated = (self.compute_tracking_error(state) > 3.0) and (self.compute_orientation_error(state) > 0.1)

	    # === Compute reward components ===    
    	# Positional and orientation errors
        pos_error = self.compute_tracking_error(state)
        orientation_error = self.compute_orientation_error(state)

    	# Control effort penalty
        control_penalty = v_left**2 + v_right**2

   	    # Drift penalty beyond a max tolerance
        # max_drift = 0.3
        # drift_penalty = max(0.0, abs(lateral_error) - max_drift) ** 2

    	# === Total reward ===
        reward = (
        	- 5.0 * pos_error
            - 5.0 * orientation_error
        	- 0.75 * control_penalty
    	)

        info = {
                "goal_met": goal_met,
                "cost": self.path_length,
                "state": state,
                "pos_error": pos_error,
                "angle_error": orientation_error,
                'control_penalty': control_penalty,
            }

        return self.getState(), reward, done, is_terminated, info
