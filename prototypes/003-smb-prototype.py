# Jesse Carleton CM3070
# initial prototype
# this file will allow for extremely basic model training for SMB
# can be used for very rudimentary training


import os
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
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

# define training checkpoint dir
checkpoint_dir = './training/'
# define logging dir
logging_dir = './log/'


# create a callback, used during training at a set interval
# saves model checkpoints
class trainingCallback(BaseCallback):
    def __init__(self, interval, filepath, verbosity=1):
        super(trainingCallback, self).__init__(verbosity)
        self.interval = interval
        self.filepath = filepath

    def _init_callback(self):
        if self.filepath is not None:
            os.makedirs(self.filepath, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.interval == 0:
            model_path = os.path.join(self.filepath, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True


# create model using a given policy
# options are CNNPolicy, MlpPolicy - convoluted nn, multi-layered perceptron
model = PPO('CnnPolicy', smb_env, verbose=1, tensorboard_log=logging_dir, learning_rate=0.00001, n_steps=512)
# get learning!
model.learn(total_timesteps=100000, callback=trainingCallback)
