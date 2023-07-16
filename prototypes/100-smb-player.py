import os
import time
import gym_super_mario_bros
import gym
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3 import PPO

class CustomReward(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 1000.0
            elif info['coins'] > 0:
                reward += info['coins'] * 100.0
            elif info['score'] > 0:
                reward += info['score'] * 1.0
            elif info['status'] == 'tall':
                reward += 100.0
            elif info['status'] == 'fireball':
                reward += 200.0
            elif info['x_pos'] > 0:
                reward += info['x_pos'] * 5.0
            elif info['time'] > 0:
                reward -= (400 - info['time']) % 3
            else:
                reward -= 50.0
        return state, reward, done, info


# set up the environment
smb_env = gym_super_mario_bros.make('SuperMarioBros-v0')
# custom rewards
smb_env = CustomReward(smb_env)
# set up the action space
smb_env = JoypadSpace(smb_env, SIMPLE_MOVEMENT)
# make it grayscale
smb_env = GrayScaleObservation(smb_env, keep_dim=True)
# create vectorized wrapper
smb_env = DummyVecEnv([lambda: smb_env])
# create a stack of N frames
smb_env = VecFrameStack(smb_env, 4, channels_order='last')
smb_env = VecMonitor(smb_env)


model = PPO.load('./model.zip')

state = smb_env.reset()
while True:
    action, _ = model.predict(state)
    state, reward, done, info = smb_env.step(action)
    smb_env.render()
    time.sleep(0.01)
