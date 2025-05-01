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
        self.data = MjData(self.model)
        self.sim_time = 0.0

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        self.time_step = 0.001
        self.n_substeps = 10
        self.time_step *= self.n_substeps
        self.viewer = None

        # for environment
        self.control_freq = 100
        self.num_time_step = int(1.0/(self.time_step*self.control_freq))
        self.max_steps = 5000
        self.cur_step = 0
        self.path_length = 0.0
        self.goal_position = np.zeros(3)

        # for state
        self.robot_pose = np.zeros(3)
        self.robot_vel = np.zeros(3)

        # for action
        self.action = np.zeros(2)

        # reference points list
        self.reference_waypoints = []

        # state & action dimension
        self.action_dim = 2
        self.state_dim = 1+len(self.robot_pose) + len(self.robot_vel)
        self.action_space = spaces.Box(-10*np.ones(self.action_dim), 10*np.ones(self.action_dim), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.state_dim), np.inf*np.ones(self.state_dim), dtype=np.float64)

    def setReferenceWaypoints(self,reference_waypoints):
        self.reference_waypoints.clear()
        for waypoint in reference_waypoints:
            self.reference_waypoints.append(waypoint)

    def render(self, **kwargs):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)

        # clear existing markers
        self.viewer.sync()

        # draw the reference waypoints
        # self.draw_waypoints(self.model, self.data, self.reference_waypoints, self.viewer)

        # # sync viewer
        # self.viewer.sync()

    def draw_waypoints(self,model, data, waypoints, viewer_handle):
        for wp in waypoints:
            t, x, y, yaw = wp
            pos = np.array([x, y, 0.1])  # slight Z offset so spheres are visible
            size = np.array([0.05, 0.05, 0.05])  # sphere radius
            rgba = np.array([1, 0, 0, 1])  # red color, fully opaque

            mujoco.mjv_addMarker(
                viewer_handle.scn,
                model,
                data,
                pos,
                None,  # no directional vector, for sphere
                rgba,
                size,
                mjtGeom=mujoco.mjtGeom.mjGEOM_SPHERE,
            )

    def getSensor(self):
        sensor_dict = {'accelerometer':None, 'velocimeter':None, 'gyro':None}
        for sensor_name in sensor_dict.keys():
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,sensor_name)
            adr = self.model.sensor_adr[id]
            dim = self.model.sensor_dim[id]
            sensor_dict[sensor_name] = self.data.sensordata[adr:adr + dim].copy()
        return sensor_dict

    def getState(self):
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
        state = {
                'time':self.sim_time,
                'pos':pos,
                'vel':vel
                }
        return state

    def getFlattenState(self):
        state = self.getState()
        time = state['time']
        pos = state['pos']
        vel = state['vel']
        state = np.concatenate([[time], pos, vel], axis=0,dtype=np.float64)
        return state

    def reset(self,seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        self.action = np.zeros(2)
        self.robot_vel = np.zeros(3)
        self.robot_pose = np.zeros(3)
        self.cur_step = 0
        self.path_length = 0.0
        self.sim_time = 0.0
        self.reference_waypoints.clear()
        state = self.getFlattenState()
        return (state,{})

    def compute_lateral_drift(self,x,y,theta,path_points):
        if len(path_points) == 0:
            return 0.0,[0.0, 0.0,0.0,0.0]
        elif len(path_points) < 2:
            nearest_pt = path_points[0]
            drift = np.linalg.norm([x-nearest_pt[0],y-nearest_pt[1]])
            return drift, nearest_pt

        tree = KDTree([p[:2] for p in path_points])
        dist, idx = tree.query([x, y])
        nearest_pt = path_points[idx]

        yaw = nearest_pt[2]
        path_dir = np.array([np.cos(yaw), np.sin(yaw)])

        robot_vec = np.array([x, y]) - np.array(nearest_pt[:2])

        lateral_error = np.cross(np.append(path_dir,0), np.append(robot_vec,0))[2]
        return lateral_error, nearest_pt

    def get_reference_point(self, state):
        """
            Get the closest reference trajectory point and its heading direction. 
            Args:
            state (np.array): Current robot position [t, x, y, theta]

            Returns:
            (t_ref, x_ref, y_ref, theta_ref): Reference position and orientation (yaw)
        """

    	# Assume self.path_points is a list of [x, y] waypoints
        if len(self.reference_waypoints) == 0:
            return state[0], state[1], state[2], state[3]
        if len(self.reference_waypoints) < 2:
            if state[0] < self.reference_waypoints[0][0]:
                return self.reference_waypoints[0][0], self.reference_waypoints[0][1], self.reference_waypoints[0][2], self.reference_waypoints[0][3]
            else:
                return state[0]+self.time_step*self.num_time_step, self.reference_waypoints[0][1], self.reference_waypoints[0][2], self.reference_waypoints[0][3]

    	# Find closest point
        future_points = [p for p in self.reference_waypoints if p[0] > state[0]]
        if future_points:
            time_diffs = [abs(p[0] - state[0]) for p in future_points]
            idx = int(np.argmin(time_diffs))
            ref_point = future_points[idx]
        else:
            future_points = [self.reference_waypoints[-1]]
            future_points[0][0] = state[0] + self.time_step*self.num_time_step
            ref_point = future_points[0]
        return ref_point[0], ref_point[1], ref_point[2], ref_point[3]

    def step(self, action):
        self.cur_step += 1
        self.sim_time += self.time_step*self.num_time_step


        prev_state = self.data.xpos[self.body_id]
        self.path_length = 0.0

        for j in range(self.num_time_step):
            mujoco.mj_forward(self.model, self.data)
            weight = 0.7
            self.action[0] = weight*self.action[0] + (1.0 - weight)*np.clip(action[0], -5.0, 5.0)
            self.action[1] = weight*self.action[1] + (1.0 - weight)*np.clip(action[1], -5.0, 5.0)

            self.data.ctrl[0] = self.action[0]
            self.data.ctrl[1] = self.action[1]
            mujoco.mj_step(self.model,self.data)
            self.path_length += np.linalg.norm(self.data.xpos[self.body_id] - prev_state)
            prev_state = self.data.xpos[self.body_id].copy()

	    # Get current state
        state = self.getFlattenState()
        pos = state[1:4]
        v_left, v_right = self.action

	    # Check termination
        goal_met = False
        if len(self.reference_waypoints) > 0:
            goal_met = np.linalg.norm(pos - self.reference_waypoints[-1][1:4]) < 0.01

        done = goal_met and (self.cur_step >= self.max_steps)
        is_terminated = self.sim_time > self.max_steps*self.time_step*self.num_time_step

	    # === Compute reward components ===

	    # Reference trajectory point (can be nearest or indexed)
        if len(self.reference_waypoints) > 0:
            t_ref, x_ref, y_ref, theta_ref = self.get_reference_point(self.getFlattenState())  # should return closest waypoint or spline sample
        else:
            t_ref, x_ref, y_ref, theta_ref,_,_,_ = self.getFlattenState()

        # Time error
        time_error = abs(self.sim_time - t_ref)
    	# Positional and orientation errors
        pos_error = np.linalg.norm(pos[1:3] - np.array([x_ref, y_ref]))
        angle_error = abs(pos[2] - theta_ref)

    	# Lateral drift
        lateral_error, _ = self.compute_lateral_drift(pos[0], pos[1], pos[2], self.reference_waypoints)

    	# Control effort penalty
        control_penalty = v_left**2 + v_right**2

   	    # Drift penalty beyond a max tolerance
        # max_drift = 0.3
        # drift_penalty = max(0.0, abs(lateral_error) - max_drift) ** 2

    	# === Total reward ===
        reward = (
            - 1.0 * time_error
        	- 5.0 * pos_error
        	- 2.0 * angle_error
        	- 0.5 * control_penalty
        	# - 5.0 * abs(lateral_error)
        	# - 10.0 * drift_penalty
    	)

        info = {
                "goal_met": goal_met,
                "cost": self.path_length,
                "state": state,
                "time_error": time_error,
                "pos_error": pos_error,
                "angle_error": angle_error,
                'control_penalty': control_penalty,
                # "lateral_error": lateral_error,
                # "drift_penalty": drift_penalty,
            }

        return self.getFlattenState(), reward, done, is_terminated, info
