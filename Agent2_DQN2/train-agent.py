import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from tensordict import TensorDict
# from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import os
# import gym_super_mario_bros
# import gym
import wandb
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
# from nes_py.wrappers import JoypadSpace
# from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack, VecMonitor, VecNormalize
# from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, ClipRewardEnv, NoopResetEnv
from smbneuralnet import CustomCNN
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

from wrappers import CustomReward, EpisodicLifeEnv, MaxAndSkipEnv, ProcessFrameResGS
from smbneuralnet import CustomCNN


# define custom neural net, timesteps, environment name
config = {
        "policy_type": 'CnnPolicy',
        "total_timesteps": 1000000,
        "env_name": 'SuperMarioBros-v0',
}


# # initialize wandb with project name, config, and requirements for metrics
# wandb_run = wandb.init(
#     project="smb-ai",
#     config=config,
#     sync_tensorboard=True,
#     monitor_gym=True,
#     save_code=True
# )

# define training checkpoint dir
checkpoint_dir = './training/'
# define logging dir
logging_dir = './log/'


CUSTOM_ACTIONS = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'B', 'A'],
    ['B', 'right', 'A'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['B', 'left', 'A'],
    ['down'],
    ['up'],
    ['A', 'up'],
]



# # custom reward function
# class CustomReward(gym.RewardWrapper):
#     def __init__(self, env):
#         super(CustomReward, self).__init__(env)
#         self._current_score = 0
#
#     def step(self, action):
#         state, reward, done, info = self.env.step(action)
#         self._current_score = info['score']
#         if done:
#             # reward if...
#             # reach the flag
#             if info['flag_get']:
#                 reward += 1000.0
#             # get coins
#             elif info['coins'] > 0:
#                 reward += info['coins'] * 100.0
#             # increase score
#             elif info['score'] > 0:
#                 reward += info['score'] * 1.0
#             # get mushroom
#             elif info['status'] == 'tall':
#                 reward += 100.0
#             # get fire flower
#             elif info['status'] == 'fireball':
#                 reward += 200.0
#             # move further right on screen
#             elif info['x_pos'] > 0:
#                 reward += info['x_pos']
#             # penalties
#             # time progression
#             elif info['time'] > 0:
#                 reward -= (400 - info['time']) % 3
#             # penalize inaction
#             else:
#                 reward -= 50.0
#         return state, reward, done, info


# training callback to save model checkpoints
class trainingCallback(BaseCallback):
    def __init__(self,
                interval,
                filepath,
                verbosity=1):
        super(trainingCallback, self).__init__(verbosity)
        # set an interval to save at
        self.interval = interval
        # set the file path saved to
        self.filepath = filepath

    def _init_callback(self) -> None:
        if self.filepath is not None:
            os.makedirs(self.filepath,
                        exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.interval == 0:
            model_path = os.path.join(self.filepath,
                        'model_checkpoint_{}'.format(self.n_calls))
            # save it!
            self.model.save(model_path)
            wandb.save(model_path)
        return True


# # set up the environment
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# # custom rewards
# env = CustomReward(env)
# # makes end-of-life the end-of-episode, but only reset on true game over.
# env = EpisodicLifeEnv(env)
# # set up the action space
# env = JoypadSpace(env, CUSTOM_ACTIONS)
# # reset env on noop for N steps
# env = NoopResetEnv(env, noop_max=30)
# # convert RGB to greyscale
# env = GrayScaleObservation(env, keep_dim=True)
# # resize from NES native resolution to 84x84
# env = ResizeObservation(env, (84, 84))
# # skip 4 frames and return the max of the last two
# env = MaxAndSkipEnv(env, skip=4)
# # limit reward to -1, 0, 1
# # env = ClipRewardEnv(env)
# # create vectorized wrapper
# env = DummyVecEnv([lambda: env])
# # create a stack of N frames
# env = VecFrameStack(env, 4, channels_order='last')
# # normalize observation or reward, uses moving average
# # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
# # monitor the vectorized environment
# env = VecMonitor(env)


# set up the environment
env = gym.make("SuperMarioBros-1-1-v0")
# custom rewards
env = CustomReward(env)
# makes end-of-life the end-of-episode, but only reset on true game over.
env = EpisodicLifeEnv(env)
# set up the action space
env = JoypadSpace(env, CUSTOM_ACTIONS)
env = ProcessFrameResGS(env)
env = MaxAndSkipEnv(env, skip=4)
# skip frames, return the max of last 2 frames
# env = MaxAndSkipEnv(env, skip=4)
# make it grayscale
# env = GrayScaleObservation(env, keep_dim=True)
# create vectorized wrapper
env = DummyVecEnv([lambda: env])
# create a stack of N frames
env = VecFrameStack(env, 4, channels_order='last')
# monitor the vectorized environment
env = VecMonitor(env)


# test to see if the env comes up
env.reset()
state, reward, done, info = env.step(action=0)
print(state)
