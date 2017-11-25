import gym


def main():
    env = gym.make('LunarLander-v2')
    env.reset()
    env.render()


if __name__ == '__main__':
    main()