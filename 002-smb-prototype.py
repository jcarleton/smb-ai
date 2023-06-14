# Jesse Carleton CM3070
# initial prototype
# this file will only populate a SMB game UI with Mario


import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO


# set up the environment
smb_env = gym_super_mario_bros.make('SuperMarioBros-v0')
# set up the action space
smb_env = JoypadSpace(smb_env, SIMPLE_MOVEMENT)
smb_env = GrayScaleObservation(smb_env, keep_dim=True)
smb_env = DummyVecEnv([lambda: smb_env])
smb_env = VecFrameStack(smb_env, 4, channels_order='last')

model = PPO('CnnPolicy', smb_env, verbose=1, learning_rate=0.00001, n_steps=512)
model.learn(total_timesteps=100000)
