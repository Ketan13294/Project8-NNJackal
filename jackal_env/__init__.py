# from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
# from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.envs.registration import register

register(
    id="JackalEnv-v0",
    entry_point="jackal_env.jackal_env:JackalEnv",
)