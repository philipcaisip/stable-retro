import gymnasium as gym
from gymnasium.core import ActType, ObsType

from typing import SupportsFloat

class BasicReward(
    gym.RewardWrapper[ObsType, ActType], gym.utils.RecordConstructorArgs
):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        pass
