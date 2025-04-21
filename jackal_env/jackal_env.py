from mujoco import MjModel, MjViewer, MjData 
from mujoco import viewer,MjRenderContextOffscreen
import mujoco

from scipy.spatial.transform import Rotation
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

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        self.time_step = 0.001
        self.n_substeps = 10
        self.time_step *= self.n_substeps
        self.viewer = None

        # for environment
        self.control_freq = 100
        self.num_time_step = int(1.0/(self.time_step*self.control_freq))
        self.max_steps = 10000
        self.cur_step = 0
        self.path_length = 0.0
        self.goal_position = np.zeros(3)

        # for state
        self.robot_pose = np.zeros(3)
        self.robot_vel = np.zeros(3)
        self.accel = np.zeros(2)

        # for action
        self.action = np.zeros(2)

        # state & action dimension
        self.action_dim = 2
        self.state_dim = len(self.robot_pose) + len(self.robot_vel)+len(self.accel)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.state_dim), np.inf*np.ones(self.state_dim), dtype=np.float32)

    def render(self, mode, **kwargs):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def getSensor(self):
        sensor_dict = {'accelerometer':None, 'velocimeter':None, 'gyro':None}
        for sensor_name in sensor_dict.keys():
            id = self.model.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,sensor_name)
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

        self.accel[0] = sensor_dict['accelerometer'][0]
        self.accel[1] = sensor_dict['accelerometer'][1]

        self.robot_pose = self.data.xpos[self.body_id].copy()
        robot_mat = self.data.xmat[self.body_id].copy()
        theta = Rotation.from_matrix(robot_mat).as_euler('zyx', degrees=False)[0]
        self.robot_pose[2] = theta

        pos = deepcopy(self.robot_pose)
        vel = deepcopy(self.robot_vel)
        state = {
                'pos':pos,
                'vel':vel,
                'accel':self.accel
                }
        return state

    def getFlattenState(self, state):
        pos = state['pos']
        vel = state['vel']
        accel = state['accel']
        state = np.concatenate([pos, vel,accel], axis=0)
        return state

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        self.action = np.zeros(2)
        self.robot_vel = np.zeros(2)
        self.robot_pose = np.zeros(3)
        self.accel = np.zeros(2)
        state = self.get_state()
        self.cur_step = 0
        self.path_length = 0.0
        return self.getFlattenState(state)

    def step(self, action):
        self.cur_step += 1        
        prev_state = self.data.xpos[self.body_id]
        for j in range(self.num_time_step):
            mujoco.mj_forward(self.model, self.data)
            weight = 0.8
            self.action[0] = weight*self.action[0] + (1.0 - weight)*np.clip(action[0], -1.0, 1.0)
            self.action[1] = weight*self.action[1] + (1.0 - weight)*np.clip(action[1], -1.0, 1.0)

            self.data.ctrl[0] = self.action[0]
            self.data.ctrl[1] = self.action[1]
            mujoco.mj_step(self.model,self.data)
            self.path_length += np.linalg.norm(self.data.xpos[self.body_id] - prev_state)

        state = self.getState()
        goal_met = False
        done = False
        if np.linalg.norm(state['pos'] - self.goal_position) < 0.01:
            goal_met = True
        if self.cur_step >= self.max_steps:
            done = True
        reward = -self.path_length
        info = {"goal_met":goal_met, 'cost':self.path_length, 'state':state}

        return self.getFlattenState(state), reward, done, info
