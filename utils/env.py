import gymnasium as gym

class CrossingEnvWithoutNoop(gym.Wrapper):
    def __init__(self,  *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.actions = list(self.actions)[:-4]  # convert to list and remove done/no-op action
        self.action_space = gym.spaces.Discrete(len(self.actions))

def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env = CrossingEnvWithoutNoop(env)
    env.reset(seed=seed)
    return env
