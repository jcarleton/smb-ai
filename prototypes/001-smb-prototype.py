# Jesse Carleton CM3070
# initial prototype
# this file will only populate a SMB game UI with Mario


import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# set up the environment
smb_env = gym_super_mario_bros.make('SuperMarioBros-v0')
# set up the action space
smb_env = JoypadSpace(smb_env, SIMPLE_MOVEMENT)


# run loop, will spawn a SMB UI
done = True
for step in range(50000):
    if done:
        state = smb_env.reset()
    state, reward, done, info = smb_env.step(smb_env.action_space.sample())
    smb_env.render()

smb_env.close()
