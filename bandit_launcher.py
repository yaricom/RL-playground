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
