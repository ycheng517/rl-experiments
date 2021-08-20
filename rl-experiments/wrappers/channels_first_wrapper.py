import gym
import numpy as np


class ChannelsFirstWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if self.observation_space.shape[-1] in [1, 3, 4]:
            shape = tuple(np.roll(self.observation_space.shape, 1))
            self.observation_space = gym.spaces.Box(
                self.observation_space.low.min(), 
                self.observation_space.high.max(), 
                shape=shape,
                dtype=np.uint8)
            self.env.observation_space = self.observation_space

        if self.observation_space.dtype != np.uint8:
            self.observation_space.dtype = np.dtype(np.uint8)

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)
        if obs.shape[-1] in [1, 3, 4]:
            obs = np.moveaxis(obs, -1, 0)
        return obs, rewards, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if obs.shape[-1] in [1, 3, 4]:
            obs = np.moveaxis(obs, -1, 0)
        return obs
