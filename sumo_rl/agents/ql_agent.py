import numpy as np

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class QLAgent:

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def set_q_table(self, q_table):
        self.q_table = q_table


    def act(self):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def get_q_table(self):
        return self.q_table

    def get_parseable_q_table(self):
        new_q_table = {}
        for key in self.q_table.keys():
            new_q_table[str(key)] = self.q_table[key]
        return new_q_table

    def learn(self, next_state, reward, done=False):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward

    def get_epsilon(self):
        return self.exploration.get_epsilon()

    def set_current_epsilon(self, epsilon_value):
        self.exploration.set_current_epsilon(epsilon_value)
