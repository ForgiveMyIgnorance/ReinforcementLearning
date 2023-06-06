import numpy as np

from solver_base import Solver


class Sarsa(Solver):
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        super(Sarsa, self).__init__(ncol, nrow, epsilon, alpha, gamma, n_action)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action[state]):
            if self.Q_table[state, i] == Q_max:
                a[i] = i
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
