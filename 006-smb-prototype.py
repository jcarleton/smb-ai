# Jesse Carleton CM3070
# initial prototype
# this file will allow for extremely basic model training for SMB
# can be used for building a custom network and modifying its parameters
# includes custom reward functionality via a Gym wrapper
# now using WandB for logging instrumentation and video snapshots


import os
import gym_super_mario_bros
import gym
import wandb
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
# old custom nn import
# from smbprototypecustompolicy import CustomNetwork, CustomActorCriticPolicy
# current custom nn import
from smbcustomnn001 import CustomNetwork, CustomActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

# define custom neural net, timesteps, environment name
config = {
        "policy_type": CustomActorCriticPolicy,
        "total_timesteps": 12000000,
        "env_name": 'SuperMarioBros-v0',
}

# initialize wandb with project name, config, and requirements for metrics
wandb_run = wandb.init(
    project="smb-ai",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
    settings=wandb.Settings(code_dir=".")
)


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
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 350.0
            elif info['coins'] > 0:
                reward += info['coins'] * 100.0
            elif info['status'] == 'tall':
                reward += 100.0
            elif info['status'] == 'fireball':
                reward += 200.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info

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

# set up the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')
env = VecNormalize(env, norm_obs=True, norm_reward=False)
env = VecMonitor(env)

# record video on interval
env = VecVideoRecorder(env, f"videos/{wandb_run.id}", record_video_trigger=lambda x: x % 50000 == 0, video_length=2000)
# define model and parameters
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{wandb_run.id}", learning_rate=0.00001, n_epochs=20, n_steps=512)

# learn the game!
model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[WandbCallback(
            gradient_save_freq=int(config["total_timesteps"]/20),
            model_save_path=f"models/{wandb_run.id}",
            verbose=2,
            ),
            trainingCallback(interval=config["total_timesteps"]/10, filepath=f"models/{wandb_run.id}")],)

# save the final model
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_SuperMario_2M')
model.save(PPO_path)

# evaluate the model
evaluate_policy(model, env, n_eval_episodes=10, render=True)
# terminate the WandB session
wandb_run.finish()
