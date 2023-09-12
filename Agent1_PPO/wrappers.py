# wrappers to use with Super Mario Bros
# Jesse Carleton for CM3070

import gym
import numpy as np
import collections
import cv2

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
        lives = info["life"]
        if lives < self.lives and lives > 0:
            # it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
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
        _, _, _, info = self.env.step(0)
        self.lives = info["life"]
        return obs

# frameskip n frames, return the max between the last 2 frames
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html#stable_baselines3.common.atari_wrappers.MaxAndSkipEnv
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """
        Return only every `skip`-th frame
        """
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
        """
        Clear past frame buffer and init to first obs
        """
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

#
# class ProcessFrameResGS(gym.ObservationWrapper):
#     """
#     Scale the observation space to 84x84
#     Convert from RGB to Grayscale
#     Checks if input is NES resolution
#     """
#     def __init__(self, env=None):
#         super(ProcessFrameResGS, self).__init__(env)
#         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
#
#     def observation(self, obs):
#         return ProcessFrameResGS.process(obs)
#
#     @staticmethod
#     def process(frame):
#         if frame.size == 240 * 256 * 3:
#             img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
#         else:
#             assert False, "Input is not native NES resolution!"
#         img = img[:, :, 0] * 0.300 + img[:, :, 1] * 0.575 + img[:, :, 2] * 0.120
#         resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
#         x_t = resized_screen[18:102, :]
#         x_t = np.reshape(x_t, [84, 84, 1])
#         return x_t.astype(np.uint8)


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
