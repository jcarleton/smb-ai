# Jesse Carleton CM3070

from collections import deque
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import random
import gym_super_mario_bros
import gym
import tensorflow as tf
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
import numpy as np
import collections
import cv2
from tqdm import tqdm
import os
import sys
import psutil

# todo - logging instrumentation
# import wandb
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


# reduce console logging, otherwise quite noisy
tf.get_logger().setLevel('INFO')
tf.keras.utils.disable_interactive_logging()


# general configuration
# hyperparameters, environment name, paths, etc
config = {
        "episode_timesteps": 2005,  # 2005 seems to be the length of in-game clock, roughly
        "total_episodes": 2500,
        "batch_size": 32,
        "noop_max": 30,
        "seed": 9876,  # used for kernel init
        "learning_rate": 0.005,
        "epsilon": 1,
        "epsilon_min": 0.1,
        "epsilon_decay_linear": 0.0001,
        # "epsilon_decay_linear": 0.9995,
        "epsilon_decay_greedy": (1/(100000*10)) * 5,
        # "epsilon_decay_greedy": 0.000005,
        "gamma": 0.99,
        "replay_memory": 100000,
        "env_name": 'SuperMarioBros-v0',
        "save_path": './artifacts/',
        "load_path": './artifacts/',
        "log_dir": './logs/'
}


# todo - wandb init
# wandb_run = wandb.init(
#     project="smb-ai-dqn",
#     sync_tensorboard=True,
#     monitor_gym=True,
#     save_code=True
# )


# custom action space for agent
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


# custom reward function
# learned implementation from https://archive.li/Nq7vC#selection-2315.0-2315.2
class CustomReward(gym.RewardWrapper):
    """
    Custom Rewards for Super Mario Bros
    """
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
                reward += 2500.0
            # get coins
            elif info['coins'] > 0:
                reward += info['coins'] * 100.0
            # increase score
            elif info['score'] > 0:
                reward += info['score']
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
            # loss of life
            elif info['life'] == 1:
                reward -= 500.0
            elif info['life'] == 0:
                reward -= 500.0
            # penalize inaction for each step
            else:
                reward -= 10.0
        return state, reward, done, info


# makes each life 1 episode, resets on game-over
# customized, but from the standard gym wrappers
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html#stable_baselines3.common.atari_wrappers.EpisodicLifeEnv
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        Customized for Super Mario Bros
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        # customized for SMB
        lives = info["life"]
        if (lives < self.lives and lives > 0) or lives == 255:
            # it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            self.was_real_done = done
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        return obs


# frameskip n frames, return the max between the last 2 frames
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html#stable_baselines3.common.atari_wrappers.MaxAndSkipEnv
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


# resize observation space (screen) to 84 x 84 and convert RGB to grayscale
# https://github.com/openai/large-scale-curiosity/blob/e0a698676d19307a095cd4ac1991c4e4e70e56fb/wrappers.py#L53
class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


# DQN Agent
# todo - complete logging functionality, collect in wandb
class MarioAgent:
    def __init__(self, state_size, action_size):
        # agent variables
        self.state_space = state_size
        self.action_space = action_size
        self.optimizer = Adam(learning_rate=config["learning_rate"], epsilon=0.01, clipnorm=1)
        self.loss_function = "Huber"
        self.acc = 0
        self.loss = 0
        self.initializer = tf.keras.initializers.HeNormal(seed=config["seed"])
        self.memory = deque(maxlen=config["replay_memory"])
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay_linear = config["epsilon_decay_linear"]
        self.epsilon_decay_greedy = config["epsilon_decay_greedy"]
        self.main_model = self.build_model()
        self.target_model = self.build_model()
        self.hard_update_target_model()
        self.soft_update_target_model()
        self.log_writer = tf.summary.create_file_writer(logdir=config["log_dir"])
        self.hash = self.hash_gen()

    # neural network architecture
    # loosely based off Nature paper (https://doi.org/10.1038/nature14236) methods section
    # replaced ReLU with SELU to fix vanishing gratient issue + dead neuron issue
    # high neuron count for GPU use only, testing currently vs simpler implementation
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, 8, strides=4, kernel_initializer=self.initializer, input_shape=self.state_space))
        # model.add(Conv2D(1024, 16, kernel_initializer=self.initializer, strides=4, input_shape=self.state_space))
        model.add(Activation("selu"))
        model.add(Conv2D(64, 4, kernel_initializer=self.initializer, strides=2))
        # model.add(Conv2D(2048, 8, kernel_initializer=self.initializer, strides=2))
        model.add(Activation("selu"))
        model.add(Conv2D(64, 3, kernel_initializer=self.initializer, strides=1))
        model.add(Conv2D(64, 3, kernel_initializer=self.initializer, strides=1))
        model.add(Activation("selu"))
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer=self.initializer, activation="selu"))
        # model.add(Dense(8192, kernel_initializer=self.initializer, activation="selu"))
        model.add(Dense(self.action_space, activation="linear"))
        # compile the model
        model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=["accuracy"])
        # plot it? looks pretty I guess
        # utils.plot_model(model, show_shapes=True)
        return model

    # update the target network
    def hard_update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    # perform soft update
    # not sure if this works well, but paper says it does
    # https://arxiv.org/pdf/1509.02971.pdf
    # implementation found at https://pylessons.com/CartPole-PER-CNN#:~:text=if%20self.,(target_model_theta)
    def soft_update_target_model(self):
        q_model_theta = self.main_model.get_weights()
        target_model_theta = self.target_model.get_weights()
        count = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = target_weight * (1 - 0.1) + q_weight * 0.1
            target_model_theta[count] = target_weight
            count += 1
        self.target_model.set_weights(target_model_theta)

    # store state transition - remember
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # choose an action
    def act(self, state):
        random_number = np.random.rand()
        if random_number <= self.epsilon:
            return np.random.randint(self.action_space)
        q_values = self.main_model.predict(state)
        q_max = np.argmax(q_values[0])
        return q_max

    # update epsilon
    # todo - capture epsilon in metrics
    def update_epsilon(self, mode, episode):
        if mode == "linear":
            if self.epsilon >= self.epsilon_min:
                self.epsilon -= self.epsilon_decay_linear
                # self.epsilon *= self.epsilon_decay_linear
        if mode == "greedy":
            self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay_greedy * episode)

    # train the network with experience replay
    def train(self, batch_size):
        # create minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.main_model.predict(state)
            if not done:
                target[0][action] = reward + self.gamma * np.amax(self.target_model.predict(next_state))
                # target[0][action] = reward + (1 - done) * self.gamma * np.amax(self.target_model.predict(next_state))
            target[0][action] = reward
            # todo - add metric capture here?
            self.main_model.fit(state, target, epochs=1, use_multiprocessing=True, verbose=1)
            self.loss, self.acc = self.main_model.evaluate(state, target, verbose=2)

    # did he die tho?
    # check if agent fell down a hole
    def agent_pitfall(self):
        y_pos = self.get_agent_y_pos()
        # true ground is 79
        if y_pos <= 50:
            return True
        else:
            return False

    # get the y position of agent
    def get_agent_y_pos(self):
        y_pos = info["y_pos"]
        return y_pos

    # metrics logging function
    # example code was seen in CM3020 week 4, 4111_code_pack (keras_io_dqn_save_weights_v1.py)
    # todo - add more metrics
    def log(self, mer, mel, rew, len, episode, epsilon, loss, accuracy, flag, score, coins, max_rew, high_score, tensorboard_log=True):
        """
        log MER, MEL, episode rewards, episode length, epsilon, loss, accuracy, goal, in game score, coins
        """
        if tensorboard_log:
            with self.log_writer.as_default():
                tf.summary.scalar("mean episode reward", mer, step=episode)
                tf.summary.scalar("mean episode length", mel, step=episode)
                tf.summary.scalar("episode reward", rew, step=episode)
                tf.summary.scalar("episode length", len, step=episode)
                tf.summary.scalar("epsilon", epsilon, step=episode)
                tf.summary.scalar("loss", loss, step=episode)
                tf.summary.scalar("accuracy", accuracy, step=episode)
                tf.summary.scalar("goal", flag, step=episode)
                tf.summary.scalar("score", score, step=episode)
                tf.summary.scalar("coins", coins, step=episode)
                tf.summary.scalar("ep reward max", max_rew, step=episode)
                tf.summary.scalar("high score", high_score, step=episode)
        # todo - different file writer for metrics... csv?
        # else:
        # with open...

    # create a random string of 8 alpha-num characters
    # guidance on this from https://pynative.com/python-generate-random-string/
    def hash_gen(self):
        hash = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(8))
        # print(hash)
        return hash

    # todo - add model loader and replay - probably another module or refactor to allow for modality
    # load a model
    def load(self, filename, type):
        if type == "main":
            file_input = config["load_path"]+filename
            load_model(self.main_model, file_input)
        if type == "target":
            file_input = config["load_path"]+filename
            load_model(self.target_model, file_input)

    # save a model
    def save(self, filename, type):
        if type == "main":
            file_output = config["save_path"]+filename+"-main-"+self.hash+".keras"
            save_model(self.main_model, file_output)
            print(f"main model being saved as {file_output}")
        if type == "target":
            file_output = config["save_path"]+filename+"-target-"+self.hash+".keras"
            save_model(self.target_model, file_output)
            print(f"target model being saved as {file_output}")



# checks if dirs exist to write to
# creates if they don't exist
def check_dirs(_path):
    if _path is not None:
        os.makedirs(_path, exist_ok=True)

# validates an episode
def episode_test(_done, _rew, _ts):
    if _done and _rew == 0.0 and _ts == 0:
        return False
    if _done and _ts < 30:
        return False
    if not _done and _ts > 30:
        return True


# set up the environment and wrap it
env = gym_super_mario_bros.make(config["env_name"])
# poor performance
# env = JoypadSpace(env, RIGHT_ONLY)
# acceptable performance
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# poor performance
# env = JoypadSpace(env, CUSTOM_ACTIONS)
env = CustomReward(env)
env = EpisodicLifeEnv(env)
env = MaxAndSkipEnv(env, 4)
env = ProcessFrame84(env)


# configure some hyperparameters
# define action/state space dimensions
# create agent
action_space = env.action_space.n
state_space = (84, 84, 1)
num_episodes = config["total_episodes"]
num_timesteps = config["episode_timesteps"]
batch_size = config["batch_size"]
dqn_agent = MarioAgent(state_space, action_space)
episode = 0
flags_got = 0
max_rew = 0
high_score = 0
reward_buffer = 0
length_buffer = 0
done = False

# check dirs, create if they don't exist
dirs = [config["save_path"], config["log_dir"]]
for n in range(len(dirs)):
    check_dirs(dirs[n])

# todo - check for models, load if avail
# run the training
while True:
    # configure some helper variables for each episode
    # agent's initial y position is == 79
    time_step = 0
    # ep_rew = []
    ep_rew = 0
    # epsilon_mean = []
    ts_done = 0
    noop_max = config["noop_max"]
    noop_action = 0

    # environment reset
    state = env.reset()
    env.reset()
    # resize for neural network
    state = state.reshape(-1, 84, 84, 1)

    # action buffer, used to validate if agent is doing noops, measured in x_pos
    action_buffer = collections.deque(maxlen=noop_max)
    action_buffer.clear()

    # run the following during each step within a given episode
    for step in tqdm(range(num_timesteps)):
        # prevent stepping into a done state
        # causes crash if not used with wrappers
        if done:
            done = False
            env.reset()
            break

        # render UI, if you want to watch
        # env.render()

        # choose an action
        action = dqn_agent.act(state)

        # pass the action to the step function
        next_state, reward, done, info = env.step(action)

        # test if Mario fell below the ground, pointless to act here
        # todo - test penalties for this to make agent learn to not die by pits
        # if dqn_agent.agent_pitfall():
        #     # print(f"he dead!")
        #     # env.reset()
        #     # done = True
        #     break

        # logic to ensure agent does not sit still for too long
        action_buffer.append(info["x_pos"])
        # print(f"action buffer is {len(action_buffer)} long")
        if len(action_buffer) == noop_max:
            buffer_check = all(_act == action_buffer[0] for _act in action_buffer)
            if buffer_check == True:
                # print(f"noop rewards before {np.sum(ep_rew)}")
                reward -= 100.0
                # ep_rew.append(reward)
                ep_rew -= 100.0
                # print(f"finally got noop! reset now! Adding penalty of -100 :: {np.sum(ep_rew)}")
                action_buffer.clear()

        # reshape the next state to pass to neural network later
        next_state = next_state.reshape(-1, 84, 84, 1)

        # metrics collection
        ts_done = step + 1
        ep_rew += reward

        if info["flag_get"]:
            flags_got += 1
            print(f"episode {episode} got flag!!!")

        if done and ep_rew == 0.0 and ts_done <= 10:
            print(f"1 something weird going on with this episode, got {ts_done} timesteps, {np.sum(ep_rew)} rewards?! Discarding...")
            env.reset()
            break
        # some are done with less than 10 timesteps, which is impossible...
        elif done and ts_done <= 10:
            print(f"2 something weird going on with this episode, got {ts_done} timesteps, {np.sum(ep_rew)} rewards?! Discarding...")
            env.reset()
            break
        else:
            # remember state transition
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state

    # # same logic as above, but continue and don't count that episode
    if done and ep_rew == 0.0 and ts_done <= 10:
        print(f"3 something weird going on with this episode, got {ts_done} timesteps, {np.sum(ep_rew)} rewards?! Discarding...")
        env.reset()
        continue
    elif done and ts_done <= 10:
        print(f"4 something weird going on with this episode, got {ts_done} timesteps, {np.sum(ep_rew)} rewards?! Discarding...")
        env.reset()
        continue
    else:
        if ep_rew == 0.0:
            print("bad state... discard...")
            env.reset()
            continue
        else:
            reward_buffer += ep_rew
            mer = reward_buffer / (episode + 1)
            length_buffer += ts_done
            mel = length_buffer / (episode + 1)
            if ep_rew > max_rew:
                max_rew = ep_rew
            if info['score'] > high_score:
                high_score = info['score']
            dqn_agent.log(mer, mel, ep_rew, length_buffer, episode+1, dqn_agent.epsilon, dqn_agent.loss, dqn_agent.acc,
                          flags_got, info['score'], info['coins'], max_rew, high_score, tensorboard_log=True)

            if len(dqn_agent.memory) > batch_size and ts_done >= 10:
                # print out stats for the run and cumulative stats
                print(f"#########################################"
                      f"episode {episode} completed! \r\n"
                      f"{ts_done} timesteps done! \r\n"
                      f"REW of {ep_rew} \r\n"
                      f"epsilon {str(dqn_agent.epsilon)[:6]} \r\n"
                      f"MER {str(mer)[:6]} \r\n"
                      f"MEL {int(mel)} \r\n"
                      f"total flags {flags_got} \r\n"
                      f"in game ep score {info['score']} \r\n"
                      f"high score {high_score} \r\n"
                      f"max rewards {max_rew} \r\n"
                      f"{info['coins']} coins got this ep \r\n"
                      f"got up to x pos {info['x_pos']}"
                      f"#########################################")

                if episode % 10 == 0:
                    print(f"mem buffer usage is {sys.getsizeof((dqn_agent.memory).copy())}")
                    print(f"process memory usage is {str(psutil.virtual_memory().used // 1e6)}")


                print(f"train model...")
                dqn_agent.train(batch_size)
                print(f"loss :: {dqn_agent.loss}, accuracy :: {dqn_agent.acc}")

                # save the models after 250 episodes until complete
                if episode % 250 == 0:
                    dqn_agent.save("dqn-mario", "main")
                    dqn_agent.save("dqn-mario", "target")

                if episode % 100 == 0:
                    if not episode == 0:
                        print(f"soft update...")
                        dqn_agent.soft_update_target_model()


            dqn_agent.update_epsilon("linear", episode)
            episode += 1

    # a way out of the infinite loop
    if episode == num_episodes:
        print(f"Training run complete! Done {num_episodes} episodes of {num_timesteps} steps!")
        break

env.close()
