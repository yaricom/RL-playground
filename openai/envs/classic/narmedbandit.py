"""
The classic N-Armed bandit environment (not bounded)
"""
import gym
from gym import spaces
import numpy as np

class NArmedBanditEnv(gym.Env):

    def __init__(self, arms, type):
        self.type = type
        self.arms = arms

        # The true action values
        self.true_action_vals = np.random.randn(self.arms)

        self.action_space = spaces.Discrete(arms)
        self.observation_space = None # no observations given

    def _reset(self):
        # Reset steps counter
        self.step_counter = 0

        return None

    def _step(self, action):
        if self.step_counter != 0:
            if self.type == 'stationary':
                # stationary - true action value + noise
                reward = self.true_action_vals[action] + np.random.standard_normal()
            else:
                # non stationary - random walk per action
                self.true_action_vals = [self.true_action_vals[i] + np.random.standard_normal() for i in range(self.arms)]
                reward = self.true_action_vals[action]
        else:
            reward = 0

        self.step_counter += 1
        return None, reward, False, {}
