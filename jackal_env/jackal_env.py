from mujoco import MjModel, MjData 
from mujoco import viewer
import mujoco
import math
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
    def __init__(self,wt = 1.0, wc = 1.0,wv = 1.0,wl = 1.0):
        abs_path = os.path.dirname(__file__)
        np.random.seed(42)
        self.model = MjModel.from_xml_path(f'{abs_path}/jackal.xml')
        self.model.opt.timestep = 0.001
        self.data = MjData(self.model)
        self.sim_time = 0.0

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        self.time_step = 0.01
        self.n_substeps = 10
        self.viewer = None
        self.scene = None

        # for environment
        self.control_freq = 100.0
        self.num_time_step = (1.0/(self.time_step*self.control_freq))
        self.max_steps = 600
        self.cur_step = 0
        self.path_length = 0.0

        # for state
        self.robot_pose = np.zeros(4)
        self.robot_vel = np.zeros(3)

        # Trajectory tracking state encoding
        self.horizon = 10
        self.max_lateral_error = 0.5
        self.encoding_dim = self.horizon + len(self.robot_vel)

        # for action
        self.action = np.zeros(2)

        # state & action dimension
        self.action_dim = 2
        #Waypoints(4 each), robot velocities (linear and angular), robot properties (length, width, wheel diameter)
        # 4*10 + 2 + 3 = 45
        self.state_dim = 45
        self.action_space = spaces.Box(-10*np.ones(self.action_dim), 10*np.ones(self.action_dim), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.state_dim), np.inf*np.ones(self.state_dim), dtype=np.float64)

        # reference points list
        self.reference_waypoints = []

        # reward weights
        self.wt = wt
        self.wc = wc
        self.wv = wv
        self.wl = wl


    def reset(self,seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.action = np.zeros(2)
        self.robot_vel = np.zeros(3)
        self.robot_pose = np.zeros(3)
        self.cur_step = 0
        self.path_length = 0.0
        self.sim_time = 0.0
        self.reference_waypoints = np.empty((0,4))

        if options is not None and 'waypoints' in options:
            self.setTrajectoryWaypoints(options['waypoints'])
        else:
            self.reference_waypoints = deepcopy(np.array(self.generate_random_waypoints(state=self.getStandardState())))

        # Set the initial position of the robot
        if self.scene is not None:
            self.scene.ngeom = len(self.reference_waypoints)
            for i in range(len(self.reference_waypoints)-1):
                mujoco.mjv_initGeom(self.scene.geoms[i],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.array([1.0,0.0,0.0,1.0]))
                mujoco.mjv_connector(self.scene.geoms[i],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008,
                        np.array([self.reference_waypoints[i][1],self.reference_waypoints[i][2],0.1]),
                        np.array([self.reference_waypoints[i+1][1],self.reference_waypoints[i+1][2],0.1]))
        state = self.getState()
        return (state,{})

    def render(self, **kwargs):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
            self.scene = self.viewer.user_scn

        # clear existing markers
        self.viewer.sync()

    def generate_random_waypoints(self,state=np.zeros((7,1)) ,duration=5.5, dt=0.01, max_lin_vel=1.0, max_ang_vel=0.5):
        n_points = int(duration / dt)+1
        t_query = np.linspace(state[0], state[0]+duration, n_points)

        lin_vel = np.clip(np.random.normal(0.0,0.5)*np.ones(n_points),-max_lin_vel,max_lin_vel) #np.clip(np.random.normal(0.0, 0.5, n_points), -max_lin_vel, max_lin_vel)
        ang_vel = 0*np.clip(np.random.normal(0.0, 0.2, n_points), -max_ang_vel, max_ang_vel) #np.clip(np.random.normal(0.0,0.1)*np.ones(n_points),-max_ang_vel,max_ang_vel) #np.clip(np.random.normal(0.0, 0.5, n_points), -max_ang_vel, max_ang_vel)

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
        yaw_ = ((yaw + math.pi)%(2*math.pi))-math.pi
        yaw = yaw_.tolist()
        waypoints = np.array([list(item) for item in zip(t_query, x, y, yaw)])
        return waypoints
    
    def setTrajectoryWaypoints(self, waypoints):
        """
        Set the trajectory waypoints for the robot.
        Args:
            waypoints (list): List of waypoints in the format [t, x, y, theta].
        """

        self.reference_waypoints = np.empty((0,4))
        self.reference_waypoints = deepcopy(np.array(waypoints))
        timestep_col = np.arange(self.cur_step*self.time_step, 
                                 self.cur_step*self.time_step+(self.reference_waypoints.shape[0]-1)*self.time_step, 
                                 self.time_step).reshape(-1, 1)
        timestep_col = np.append(timestep_col,np.array([timestep_col[-1][0] + self.time_step])).reshape(-1,1)
        self.reference_waypoints = deepcopy(np.hstack((timestep_col, self.reference_waypoints)))
        if self.scene is not None:
            self.scene.ngeom = len(self.reference_waypoints)
            for i in range(len(self.reference_waypoints)-1):
                mujoco.mjv_initGeom(self.scene.geoms[i],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.array([1.0,0.0,0.0,1.0]))
                mujoco.mjv_connector(self.scene.geoms[i],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, 0.05,
                        np.array([self.reference_waypoints[i][1],self.reference_waypoints[i][2],0.1]),
                        np.array([self.reference_waypoints[i+1][1],self.reference_waypoints[i+1][2],0.1]))

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

        pos = self.robot_pose.copy()
        vel = self.robot_vel.copy()
        state = np.concatenate([np.array([self.sim_time]), pos, vel])
        return state
    
    def normalize_angle(self,angle):
        return ((angle+math.pi)/(2*math.pi))-math.pi
    
    def getState(self):
        state = self.getStandardState()
        z_t = self.get_reference_point(state)
        # Robot Length, Robot Width, Wheel Diameter
        o_d = [0.508,0.43,0.1]
        e_state = np.concatenate([z_t, [np.linalg.norm(state[4:6]),state[6]],o_d], axis=0,dtype=np.float64)
        return e_state

    def get_reference_point(self, state):
        """
            Get the horizon length trajectory segment from the current time step
            Args:
            state (np.array): Current robot position [t, x, y, theta]

            Returns:
            ref_points: Segment of the trajectory of horizon STEPS
        """        
        future_points = [deepcopy(p) for p in self.reference_waypoints if p[0] > state[0]]
        # If no future points are found, use the current position of the robot but with zero orientation
        if len(future_points) == 0:
            future_points = np.tile([state[0],state[1],state[2],state[3]],(self.horizon,1))
            for i in range(self.horizon):
                future_points[i][0] = state[0] + self.time_step*self.num_time_step*(i+1) 
            ref_point = future_points

        # If there are future points but not enough to fill the horizon, fill the rest with the last point
        elif len(future_points)  < self.horizon:
            ref_point = future_points
            dt = self.time_step*self.num_time_step
            for _ in range(self.horizon - len(ref_point)):
                ref_point.append(ref_point[-1].copy())
                ref_point[-1][0] += dt
        
        # If there are enough future points, select horizon points
        else:
            time_diffs = [abs(p[0] - state[0]) for p in future_points]
            idx = int(np.argmin(time_diffs))
            ref_point = future_points[idx:idx + int(self.horizon)]

        R = Rotation.from_euler('z', -state[3], degrees=False).as_matrix()[:2,:2]
        Rt = -R@(state[1:3])
        for i in  range(len(ref_point)):
            ref_point[i][0] = ref_point[i][0] - state[0]
            ref_point[i][1:3] = (R@(ref_point[i][1:3]) + Rt)
            ref_point[i][3] = (ref_point[i][3]) - (state[3])
        
        ref_point = np.array(ref_point).flatten(order='F')
        return ref_point


    def compute_tracking_error(self,state):
        nearest_pt = next((p for p in self.reference_waypoints if p[0] >= state[0]), None)
        if nearest_pt is None:
            return 0.0,0.0,0.0
        return np.linalg.norm(state[1:3] - nearest_pt[1:3]),abs(state[3] - nearest_pt[3]), np.linalg.norm(state[4:7])
        
    def calculate_distance_to_goal(self, state, wp=1.0, wtheta=1.0, wv=2.5):
        """
        Calculate combined distance to goal including position, orientation, and velocities.
        
        state: [t, x, y, theta, xdot, ydot, thetadot]
        reference_waypoints: list or array of [t, x, y, theta]
        """
        goal = self.reference_waypoints[-1]  # final waypoint
        dx = state[1] - goal[1]
        dy = state[2] - goal[2]
        dpos = np.hypot(dx, dy)
        
        dtheta = abs(((state[3] - goal[3] + np.pi) % (2*np.pi)) - np.pi)  # wrap to [-pi, pi]
        
        v_norm = np.linalg.norm(state[4:7])  # speed magnitude (should stop)
        
        if (dpos<=1.0):
            distance = (wp * dpos + wtheta * dtheta + wv * v_norm)
        else:
            distance = 0.0
        return distance
        
    def calculate_lateral_error(self, state):
        """
        Calculate lateral error (signed distance) to the nearest waypoint direction.
        """
        # Find the closest waypoint in time
        future_waypoints = [wp for wp in self.reference_waypoints if wp[0] >= state[0]]
        if not future_waypoints:
            nearest_wp = self.reference_waypoints[-1]
        else:
            nearest_wp = future_waypoints[0]
        
        x, y = state[1], state[2]
        x_r, y_r, theta_r = nearest_wp[1], nearest_wp[2], nearest_wp[3]
        
        dx = x - x_r
        dy = y - y_r
    
        # Compute lateral error: projection onto normal vector
        lateral_error = dx * -np.sin(theta_r) + dy * np.cos(theta_r)
        return lateral_error

    def step(self, action):
        self.cur_step += 1
        self.sim_time += self.time_step * self.num_time_step
        self.sim_time = round(self.sim_time, 4)

        prev_state = self.data.xpos[self.body_id]
        self.path_length = 0.0
        weight = 0.3
        
        for j in range(self.n_substeps):
            mujoco.mj_forward(self.model, self.data)

            self.action[0] = weight*self.action[0] + (1.0 - weight)*np.clip(action[0], -2, 2)
            self.action[1] = weight*self.action[1] + (1.0 - weight)*np.clip(action[1], -2, 2)

            self.data.ctrl[0] = self.action[0]
            self.data.ctrl[1] = self.action[1]
            mujoco.mj_step(self.model,self.data)
            self.path_length += np.linalg.norm(self.data.xpos[self.body_id] - prev_state)
            prev_state = self.data.xpos[self.body_id].copy()

	    # Get current state
        state = self.getStandardState()
        v_left =  (self.action[0] + 2.0)/4.0
        v_right = (self.action[1] + 1.5)/4.0

	    # Check termination and done conditions
        #goal_met = (np.linalg.norm(state[1:4] - self.reference_waypoints[min(self.max_steps-1,len(self.reference_waypoints)-1)][1:4]) < 0.01 and np.linalg.norm(state[4:7])<0.01)
        goal_met = (
            (np.linalg.norm(state[1:3]-self.reference_waypoints[min(self.max_steps-1,len(self.reference_waypoints)-1)][1:3]) < 0.01) and
            (np.linalg.norm(state[4:6]) < 0.001) and
            (abs(state[3]-self.reference_waypoints[min(self.max_steps-1,len(self.reference_waypoints)-1)][3]) < 0.01) and
            (abs(state[6]) < 0.001)
        )

        done = goal_met or (self.cur_step >= self.max_steps-1)

	    # === Compute reward components ===    
    	# Positional and orientation errors
        tracking_error,orientation_error, velocity_norm = self.compute_tracking_error(state)

    	# Control effort penalty
        control_penalty = v_left**2 + v_right**2
        lateral_penalty = self.calculate_lateral_error(state)

        is_terminated = False

    	# === Total reward ===
        reward = (
            - 0.01    * self.cur_step
        	- self.wt * tracking_error
            - self.wt * orientation_error
		    - self.wv * velocity_norm
        	- self.wc * control_penalty
        	- self.wl * lateral_penalty)
    	
        reward+=20.0*np.exp(-5.0 * self.calculate_distance_to_goal(state))

        info = {
                "goal_met": goal_met,
                "cost": self.path_length,
                "state": state,
                "velocity_norm":velocity_norm,
                "pos_error": tracking_error,
                "angle_error": orientation_error,
                "control_penalty": control_penalty,
                "lateral_penalty":lateral_penalty,
                "distance_to_goal":self.calculate_distance_to_goal(state)
                
            }

        return self.getState(), reward, done, is_terminated, info
