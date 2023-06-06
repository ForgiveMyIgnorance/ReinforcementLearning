import time

from cliff_walking_env import CliffWalkingEnv
from dyna_Q import DynaQ

import numpy as np
import random
import matplotlib.pyplot as plt


# 下面是算法在环境中的训练函数，输入参数是Q-planning的步数
def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha,gamma, n_planning)
    num_episodes = 300

    return_list = []
    for i in range(num_episodes):
        epsilon_return = 0
        state = env.reset()

        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            epsilon_return += reward  # 这里的回报计算不进行gamma衰减
            agent.update(state, action,reward, next_state)
            state = next_state
        return_list.append(epsilon_return)
    return return_list


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20]
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list, label=str(n_planning) + ' planning steps')
    plt.legend()
    plt.xlabel('Epsilon')
    plt.ylabel('Returns')
    # plt.title('Sarsa on {}'.format('CliffWalkingProblem'))
    plt.title('Dyna-Q on {}'.format('CliffWalkingProblem'))
    plt.show()
    