import gymnasium as gym
import numpy as np
import retro
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecEnv
)

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    # env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


if __name__ == "__main__":

    def make_env():
        env = make_retro(game="Airstriker-Genesis") 
        env = wrap_deepmind_retro(env)
        return env

    env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))

    model = PPO.load("scratch/model.zip", env)
    # model.set_env(env)
    
    obs = model.get_env().reset()

    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
    while True:
        if state is not None:
            actions = model.predict(obs,S=state, M=dones)
        else:
            actions = model.predict(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('episode_rew={}'.format(episode_rew[i]))
                episode_rew[i] = 0
                
    env.close()

