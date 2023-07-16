# Jesse Carleton CM3070
# initial prototype
# this file will allow for extremely basic model training for SMB
# can be used for building a custom network and modifying its parameters
# includes custom reward functionality via a Gym wrapper, now in use
# now using WandB for logging instrumentation and video snapshots


import os
import gym_super_mario_bros
import gym
import wandb
import cv2
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, FrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from smbcustomnn001 import CustomNetwork, CustomActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

# define custom neural net, timesteps, environment name
config = {
        "policy_type": CustomActorCriticPolicy,
        "total_timesteps": 100000,
        "env_name": 'SuperMarioBros-v0',
}


# define training checkpoint dir
checkpoint_dir = './training/'
# define logging dir
logging_dir = './log/'

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
                reward += info['x_pos']
            elif info['time'] > 0:
                reward -= (400 - info['time']) % 3
            else:
                reward -= 50.0
        return state, reward, done, info

class trainingCallback(BaseCallback):
    def __init__(self, interval, filepath, verbosity=1):
        super(trainingCallback, self).__init__(verbosity)
        self.interval = interval
        self.filepath = filepath
    def _init_callback(self) -> None:
        if self.filepath is not None:
            os.makedirs(self.filepath, exist_ok=True)
    def _on_step(self) -> bool:
        if self.n_calls % self.interval == 0:
            model_path = os.path.join(self.filepath, 'model_checkpoint_{}'.format(self.n_calls))
            self.model.save(model_path)
            wandb.save(model_path)
        return True


class ResizeObservation(gym.ObservationWrapper):
    """Downsample the image observation to a square image. """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation

bestModelCallback = EvalCallback(callback_on_new_best=saveModelNow())

# set up the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = CustomReward(env)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')
# normalize observation or reward, uses moving average
# env = VecNormalize(env, norm_obs=True, norm_reward=False)
env = VecMonitor(env)


steps = [512, 1024, 2048]
# epochs = [5, 10, 15, 20]
lr = [0.00001, 0.0001, 0.001, 0.01, 0.1]
ent = [0.00001, 0.0001, 0.001, 0.01]

for entco in ent:
    for lrate in lr:
        for step in steps:

            # initialize wandb with project name, config, and requirements for metrics
            wandb_run = wandb.init(
                project="smb-ai-sweep3",
                config=config,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
                settings=wandb.Settings(code_dir=".")
            )

            # record video on interval
            env = VecVideoRecorder(env, f"videos/{wandb_run.id}", record_video_trigger=lambda x: x % 50000 == 0, video_length=2000)

            # define model and parameters
            model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{wandb_run.id}", ent_coef=entco, learning_rate=lrate, n_epochs=epoch, n_steps=step)

            # learn the game!
            model.learn(
                        total_timesteps=config["total_timesteps"],
                        callback=[WandbCallback(
                        gradient_save_freq=int(config["total_timesteps"]/20),
                        model_save_path=f"models/{wandb_run.id}",
                        verbose=2,
                        ),
                        trainingCallback(interval=config["total_timesteps"]/10, filepath=f"models/{wandb_run.id}")
                        ],)

            # save the final model
            PPO_path = os.path.join('Training', 'Saved Models', f"PPO_SuperMario_1M/{wandb_run.id}")
            model.save(PPO_path)

            # evaluate the model
            evaluate_policy(model, env, n_eval_episodes=10, render=False)
            # terminate the WandB session
            wandb_run.finish()
            env.reset()
