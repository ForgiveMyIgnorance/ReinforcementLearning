import numpy as np


class Solver:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        raise NotImplementedError

    def best_action(self, state):
        raise NotImplementedError

    def update(self, s0, a0, r, s1, a1):
        raise NotImplementedError
