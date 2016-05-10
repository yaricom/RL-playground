from openai.agents.sampleaverage import SampleAverageActionValueAgent
from openai.envs.classic.narmedbandit import NArmedBanditEnv

import numpy as np
import matplotlib.pyplot as plt

def main():
    # setup
    mean_results = evaluate(epsilon = 0) # greedy
    mean_results_01 = evaluate(epsilon = 0.1)
    mean_results_001 = evaluate(epsilon = 0.01)

    #print "Results: {}".format(mean_results)

    plt.plot(mean_results, color='g', label='$\epsilon$ = 0 (greedy)')
    plt.plot(mean_results_01, label='$\epsilon$ = 0.1')
    plt.plot(mean_results_001, color='r', label='$\epsilon$ = 0.01')

    plt.legend(loc="lower right")
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.show()

def evaluate(epsilon, bandits_count = 2000, max_steps = 1000):
    reward = 0
    done = False

    results = np.zeros([bandits_count, max_steps])

    for i in xrange(bandits_count):
        agent = SampleAverageActionValueAgent(num_actions = 10, epsilon = epsilon)
        bandit = NArmedBanditEnv(10, 'stationary')
        bandit._reset()

        for j in xrange(max_steps):
            action = agent.evaluate(reward, done)
            ob, reward, done, _ = bandit._step(action)
            results[i, j] = reward
            if done:
                break

    return np.mean(results, axis = 0)


if __name__ == '__main__':
    main()
