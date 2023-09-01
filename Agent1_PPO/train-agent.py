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
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
# from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
from smbcustomnn001 import CustomNetwork, CustomActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
from wrappers import EpisodicLifeEnv, CustomReward, MaxAndSkipEnv, ProcessFrameResGS
from stable_baselines3.common.logger import TensorBoardOutputFormat


# define custom neural net, timesteps, environment name
config = {
        "policy_type": CustomActorCriticPolicy,
        "total_timesteps": 1000000,
        "env_name": 'SuperMarioBros-v0',
}


# initialize wandb with project name, config, and requirements for metrics
wandb_run = wandb.init(
    project="smb-ai-ppo",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True
)

# artifact = wandb.Artifact(name='source', type='code')
# artifact.add_file('./train-agent.py', name='train-agent.py')
# artifact.add_file('./smbcustomnn001.py', name='smbcustomnn001.py')


# define training checkpoint dir
checkpoint_dir = './training/'
# define logging dir
logging_dir = './log/'


# training callback to save model checkpoints
class trainingCallback(BaseCallback):
    def __init__(self,
                interval,
                filepath,
                verbosity=1
                ):
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


class SummaryWriterCallback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''
    def __init__(self):
        super().__init__()
        self.goals = 0
        self.ep_rew_max = 0
        self.ep_rew = 0
        self.high_score = 0
        self.high_score_counter = 0

    def _on_training_start(self):
        self._log_freq = 1  # log every 1 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        if self.n_calls % self._log_freq == 0:
            # print(f"locals is {self.locals.keys()}")
            if self.locals['n_steps'] == 0:
                self.ep_done = False
            if self.locals['n_steps'] > 0:
                self.ep_done = self.locals['done']

            # print(f"see if the ep is done.. "
            #       f"num steps = {self.locals['n_steps']}")


            score = self.locals['infos'][0]['score']
            # print(f"got this score {score}")
            self.tb_formatter.writer.add_scalar("game/score", score, self.n_calls)

            if self.locals['infos'][0]['flag_get']:
                self.goals += 1
            for i in range(self.locals['env'].num_envs):
                self.tb_formatter.writer.add_scalar("game/goal", self.goals, self.n_calls)

            coins = self.locals['infos'][0]['coins']
            self.tb_formatter.writer.add_scalar("game/coins", coins, self.n_calls)

            if not self.ep_done:
                self.ep_rew += self.locals['rewards'][0]

            if self.ep_done:
                self.ep_rew += self.locals['rewards'][0]
                self.high_score_counter = self.locals['infos'][0]['score']

                if self.ep_rew > self.ep_rew_max:
                    self.ep_rew_max = self.ep_rew
                    self.tb_formatter.writer.add_scalar("game/ep_rew_max", self.ep_rew_max, self.n_calls)
                    self.ep_rew = 0

                if self.high_score_counter > self.high_score:
                    self.high_score = self.high_score_counter
                    self.tb_formatter.writer.add_scalar("game/high score", self.high_score, self.n_calls)
                    self.high_score_counter = 0

                self.ep_rew = 0


# set up the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# custom rewards
env = CustomReward(env)
# makes end-of-life the end-of-episode, but only reset on true game over.
env = EpisodicLifeEnv(env)
# set up the action space
env = JoypadSpace(env, SIMPLE_MOVEMENT)
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


# record video on interval
env = VecVideoRecorder(env,
    f"videos/{wandb_run.id}",
    record_video_trigger=lambda x: x % 50000 == 0,
    video_length=2000
    )


# define model and parameters
model = PPO(config["policy_type"],
            env,
            verbose=1,
            tensorboard_log=f"runs/{wandb_run.id}",
            learning_rate=0.00001,
            n_epochs=20,
            n_steps=256,
            batch_size=256,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5
            )


# learn the game!
model.learn(total_timesteps=config["total_timesteps"],
            callback=[WandbCallback(
            gradient_save_freq=int(config["total_timesteps"]/20),
            model_save_path=f"models/{wandb_run.id}",
            verbose=2,
            ),
            trainingCallback(interval=config["total_timesteps"]/10,
            filepath=f"models/{wandb_run.id}"),
            SummaryWriterCallback()
            ],
            )


# save the final model
PPO_path = os.path.join('Training',
        'Saved Models',
        f"PPO_SuperMario_1M/{wandb_run.id}"
        )


model.save(PPO_path)

# evaluate the model
evaluate_policy(model,
                env,
                n_eval_episodes=10,
                render=False
                )

# terminate the WandB session
wandb_run.finish()
