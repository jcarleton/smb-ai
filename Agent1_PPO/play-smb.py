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


# custom reward function
class CustomReward(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._current_score = info['score']
        if done:
            # reward if...
            # reach the flag
            if info['flag_get']:
                reward += 1000.0
            # get coins
            elif info['coins'] > 0:
                reward += info['coins'] * 100.0
            # increase score
            elif info['score'] > 0:
                reward += info['score'] * 1.0
            # get mushroom
            elif info['status'] == 'tall':
                reward += 100.0
            # get fire flower
            elif info['status'] == 'fireball':
                reward += 200.0
            # move further right on screen
            elif info['x_pos'] > 0:
                reward += info['x_pos']
            # penalties
            # time progression
            elif info['time'] > 0:
                reward -= (400 - info['time']) % 3
            # penalize inaction
            else:
                reward -= 50.0
        return state, reward, done, info


# set up the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# custom rewards
env = CustomReward(env)
# set up the action space
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# make it grayscale
env = GrayScaleObservation(env, keep_dim=True)
# create vectorized wrapper
env = DummyVecEnv([lambda: env])
# create a stack of N frames
env = VecFrameStack(env, 4, channels_order='last')
# monitor the vectorized environment
env = VecMonitor(env)

# load pretrained model
model = PPO.load('./model.zip')

# initialize state
state = env.reset()

# infinite loop to play the game, render the UI
# control+c to break
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
