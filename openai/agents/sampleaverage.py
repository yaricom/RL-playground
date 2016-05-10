"""
The sample-average action-value learning agent
"""
import random as rnd
import numpy as np

class SampleAverageActionValueAgent(object):

    def __init__(self, num_actions, epsilon = 0.1, step_size = 0, learning_rate = 1, decay = 0.9):
        """Creates new instance.

        Keyword arguments:
        num_actions -- the number of actions in action space
        epsilon -- the probability on non-greedy action selection (default 0.1)
        step_size -- the constant step size for action value calculation (default 0). If
        set to default than assumed stationary environment and step size will be calculated
        at each step per rewarded action as 1/K (K - number of rewards for particular action)
        learning_rate -- the learning rate to correct best values after episode
        decay -- the learning rate decay to gurantee convergence
        """

        self.name = 'sample-average action-value agent'
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.step_size = step_size

        self.learning_rate = learning_rate
        self.decay = decay

        self.best_score = 0
        self.episode_score = 0
        self.best_action_vals = np.zeros(self.num_actions)
        self.action = np.random.random_integers(0, self.num_actions - 1) # start with random action

    def _episodeActionValues(self):
        self.action_vals = [self.best_action_vals[i] + np.random.ranf() * self.learning_rate for i in range(self.num_actions)]
        self.actions_reward_count = np.zeros(self.num_actions)

    def _nextAction(self, reward, prev_action):
        # find incremental action-value for rewarded action
        if self.step_size == 0:
            # stationary environment
            K = self.actions_reward_count[prev_action] + 1
            self.action_vals[prev_action] = self.action_vals[prev_action] + (reward - self.action_vals[prev_action]) / K # Qk + 1/k * [Rk - Qk]
            self.actions_reward_count[prev_action] = K
        else:
            raise error.Error('Non-stationary environment not supported. Please make sure that [step_size] is zero' )

        # do best next action selection E-greedy
        if np.random.ranf() > self.epsilon:
            # find action with maximal value
            best_action = np.argmax(self.action_vals)
        else:
            # select action index randomly
            best_action = np.random.random_integers(0, self.num_actions - 1)

        return best_action

    def evaluate(self, reward, done):
        """Do evaluate provided reward and return action selected for next step

        Keyword arguments:
        reward -- the reward for previous action evaluated received from environment
        done -- the flag to indicate whether current episode is done
        """

        if self.episode_score == 0:
            self._episodeActionValues()

        # Find best action
        self.action = self._nextAction(reward, self.action)

        # Update score
        self.episode_score += reward

        if done:
            if self.episode_score >= self.best_score:
                self.best_score = self.episode_score
                self.best_action_vals = self.action_vals
                self.learning_rate *= self.decay
            else:
                self.learning_rate /= self.decay

            # reset current score for next episode
            self.episode_score = 0

        return self.action
