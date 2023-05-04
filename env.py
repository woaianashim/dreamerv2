import os
import gym
import torch
import numpy as np
from gym.wrappers.clip_action import ClipAction
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.monitor import Monitor as Video
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame,)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class TransposeAndNormalizeImage(gym.ObservationWrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super().__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            -1,
            1, [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])/255.

class ToTensorEnv(gym.ObservationWrapper):
    def __init__(self, env=None, device="cpu"):
        super(ToTensorEnv, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0,  0, 0],
            self.observation_space.high[0,  0, 0], [
                obs_shape[0], obs_shape[1], obs_shape[2]
            ],
            dtype=self.observation_space.dtype)
        self.device = device

    def observation(self, observation):
        return torch.tensor(np.array(observation), device=self.device).float()#.squeeze()

def make_env(env_name, is_atari=True, record=False, noops=30, action_repeat=4,
             size=(64, 64), life_done=False, grayscale=False, device="cuda"):
    env = gym.make(env_name)
    env.seed(123)
    if record:
        print("Recording...")
        env = Video(env, "./video", force=True)
    if is_atari:
        env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale)
        #env = NoopResetEnv(env, noop_max=30)
        #env = FireResetEnv(env)
        #env = EpisodicLifeEnv(env)
        #env = MaxAndSkipEnv(env, skip=4)
        #env = WarpFrame(env, width=64, height=64)
        env = ClipRewardEnv(env)
        env = TransposeAndNormalizeImage(env)
        #env = FrameStack(env, 4)
        print(env.observation_space.shape)
    env = ToTensorEnv(env, device=device)
    return env
