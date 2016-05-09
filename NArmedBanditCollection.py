import random as rnd
import numpy as np

class NArmedBanditAgent(object):

    def __init__(self, arms = 10, steps = 1000, epsilon = 0.01, type = 'stationary'):
        self.type = type
        self.steps = steps
        self.epsilon = epsilon
        self.arms = arms

    # Reset to initial values
    def reset(self):
        # The true action values
        self.true_action_vals = np.random.randn(self.arms)
        print "True action values: {}".format(self.true_action_vals)
        # The action values for current step
        self.action_vals = self.true_action_vals
        # the best action values per step
        self.best_action_vals = np.zeros(self.steps)

    # Function to get reward value for provided action
    def _step_reward(self, action, K):
        if K == 0:
            return 0
        # find action-value incremental reward
        reward = self.true_action_vals[action] + rnd.random()
        reward = self.action_vals[action] + (reward - self.action_vals[action]) / K # Qk + 1/k * [Rk - Qk]
        return reward

    # Perform one step
    def _step(self, K):
        # calculate action rewards
        self.action_vals = [self.true_action_vals[i] + self._step_reward(i, K) for i in range(self.arms)]
        # find best action E-greedy
        if np.random.ranf() > self.epsilon:
            # find action with maximal value
            best_action = np.argmax(self.action_vals)
        else:
            # select action index randomly
            best_action = np.random.random_integers(0, self.arms - 1)
            print "+++++++++++++++++Select random action"

        self.best_action_vals[K] = self.action_vals[best_action]

    # Executes n-armed bandit simulation and returns best action rewards per step
    def execute(self):
        for i in xrange(self.steps):
            self._step(i)

        return self.best_action_vals
