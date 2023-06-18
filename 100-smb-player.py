import os
import gym_super_mario_bros
import gym
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from smbprototypecustompolicy import CustomNetwork, CustomActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy

# define training checkpoint dir
checkpoint_dir = './training/'
# define logging dir
logging_dir = './log/'


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


model = PPO.load('./train/best_model_1000')


state = smb_env.reset()
while True:

    action, _ = model.predict(state)
    state, reward, done, info = smb_env.step(action)
    smb_env.render()

