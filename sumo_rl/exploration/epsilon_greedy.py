import numpy as np


class EpsilonGreedy:

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.count = 0

    def choose(self, q_table, state, action_space):
        #print("Inside Epsilon Greedy -> Epsilon: {}".format(self.epsilon))
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = np.argmax(q_table[state])

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        self.count += 1
        #print(self.epsilon)
        return action

    def reset(self):
        self.epsilon = self.initial_epsilon

    def get_epsilon(self):
        return self.epsilon

    def set_current_epsilon(self, epsilon_value):
        self.epsilon = epsilon_value
