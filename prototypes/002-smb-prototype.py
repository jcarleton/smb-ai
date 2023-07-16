# Jesse Carleton CM3070
# initial prototype
# this file will allow for extremely basic model training for SMB
# not adivsed to be used for actual training


import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO


# set up the environment
smb_env = gym_super_mario_bros.make('SuperMarioBros-v0')
# set up the action space
smb_env = JoypadSpace(smb_env, SIMPLE_MOVEMENT)
# make it grayscale
smb_env = GrayScaleObservation(smb_env, keep_dim=True)
# create vectorized wrapper
smb_env = DummyVecEnv([lambda: smb_env])
# create a stack of N frames
smb_env = VecFrameStack(smb_env, 4, channels_order='last')


# create model using a given policy
# options are CNNPolicy, MlpPolicy - convoluted nn, multi-layered perceptron
model = PPO('CnnPolicy', smb_env, verbose=1, learning_rate=0.00001, n_steps=512)
# get learning!
model.learn(total_timesteps=100000)
