# RL-playground
In this repository some of my experiments with Reinforcement Learning algorithms based on [OpenAi Gym ToolKit](https://gym.openai.com)

## Overview
Packages:
- [openai/envs](https://github.com/yaricom/RL-playground/tree/master/openai/envs) the OpenAi Gym compatible environments for evaluation
- [openai/agents](https://github.com/yaricom/RL-playground/tree/master/openai/agents) the learning agents

Environments:
- [NArmedBanditEnv](https://github.com/yaricom/RL-playground/blob/master/openai/envs/classic/narmedbandit.py) - N-armed bandit (stationary, nonstationary)

Learning agents:
- [SampleAverageActionValueAgent](https://github.com/yaricom/RL-playground/blob/master/openai/agents/sampleaverage.py) - the learning agent based on sample-average action-value selection algorithm for both stationary and nonstationary environments

## Usage
```Python
import gym

from openai.agents.sampleaverage import SampleAverageActionValueAgent

def main():
    # load environment
    env = gym.make('10ArmedBanditStationary-v0')

    # setup
    agent = SampleAverageActionValueAgent(num_actions = 10)
    episode_count = 1
    max_steps = 100
    reward = 0
    done = False

    for i in xrange(episode_count):
        ob = env.reset()

        for j in xrange(max_steps):
            action = agent.evaluate(reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break


if __name__ == '__main__':
    main()
```

