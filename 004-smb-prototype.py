# Jesse Carleton CM3070
# initial prototype
# this file will allow for extremely basic model training for SMB
# can be used for building a custom network and modifying its parameters
# includes custom reward functionality via a Gym wrapper


import os
import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3 import PPO
from smbprototypecustompolicy import CustomNetwork, CustomActorCriticPolicy


# define training checkpoint dir
checkpoint_dir = './training/'
# define logging dir
logging_dir = './log/'

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info


# set up the environment
smb_env = gym_super_mario_bros.make('SuperMarioBros-v0')
smb_env = CustomReward(smb_env)
# set up the action space
smb_env = JoypadSpace(smb_env, SIMPLE_MOVEMENT)
# make it grayscale
smb_env = GrayScaleObservation(smb_env, keep_dim=True)
# create vectorized wrapper
smb_env = DummyVecEnv([lambda: smb_env])
# create a stack of N frames
smb_env = VecFrameStack(smb_env, 4, channels_order='last')
# smb_env = VecMonitor(smb_env, logging_dir) -- future use?
smb_env = VecNormalize(smb_env, norm_obs=True, norm_reward=True)


# create a callback, used during training at a set interval
# saves model checkpoints
class trainingCallback(BaseCallback):
    def __init__(self, interval, filepath, verbosity=1):
        super(trainingCallback, self).__init__(verbosity)
        self.interval = interval
        self.filepath = filepath
        self.unwrapped = smb_env.unwrapped
        self.observation_space = smb_env.observation_space
        self.action_space = smb_env.action_space
        self.metadata = smb_env.metadata
        # self.best_mean_reward = -np.inf
        # self.save_path = os.path.join(self.filepath, 'best_model_{}'.format(self.n_calls))
        # self.eval_env = smb_env
    def _init_callback(self) -> None:
        if self.filepath is not None:
            os.makedirs(self.filepath, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.interval == 0:
            model_path = os.path.join(self.filepath, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

            # EvalCallback(self, eval_freq=1000, verbose=1)
            # self.logger.record("reward", self.best_mr)
            # self.logger.record("reward", self.best_mean_reward)
        return True

# future use?
# callbackGroup = CallbackList([trainingCallback, nextCallback, ...])


callback = trainingCallback(interval=1000, filepath=checkpoint_dir)
# create model using a given policy
# options are CnnPolicy, MlpPolicy - convoluted nn, multi-layered perceptron
model = PPO(CustomActorCriticPolicy, smb_env, verbose=1, tensorboard_log=logging_dir, learning_rate=0.00001, n_steps=512)
# model = PPO("CnnPolicy", smb_env, verbose=1, tensorboard_log=logging_dir, learning_rate=0.00001, n_steps=512)
# get learning!
model.learn(total_timesteps=10000, callback=callback)
# evaluate_policy(model, smb_env, n_eval_episodes=10)
