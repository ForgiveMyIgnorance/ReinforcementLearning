import matplotlib.pyplot as plt

from cliff_walking_env import CliffWalkingEnv
# from sarsa import Sarsa
from q_learning import QLearning

import numpy as np


if __name__ == '__main__':
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    # agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500

    return_list = []
    for i in range(500):
        epsilon_return = 0
        state = env.reset()
        action = agent.take_action(state)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.take_action(next_state)
            epsilon_return += reward #这里的回报计算不进行gamma衰减
            agent.update(state, action,reward, next_state, next_action)
            state = next_state
            action = next_action
        return_list.append(epsilon_return)

    epsilon_list = list(range(len(return_list)))
    plt.plot(epsilon_list, return_list)
    plt.xlabel('Epsilon')
    plt.ylabel('Returns')
    # plt.title('Sarsa on {}'.format('CliffWalkingProblem'))
    plt.title('QLearning on {}'.format('CliffWalkingProblem'))
    plt.show()