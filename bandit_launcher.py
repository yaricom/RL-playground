from NArmedBanditCollection import NArmedBanditAgent

def main():
    # Main configuration
    episode_count = 2
    max_steps = 1000

    bandit = NArmedBanditAgent(arms = 10, steps = max_steps)

    for i in xrange(episode_count):
        bandit.reset()
        values = bandit.execute()
        print values


if __name__ == '__main__':
    main()
