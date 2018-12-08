# ReplayAgent.py
from collections import namedtuple
import random

import numpy as np

from Agent import QAgent

Action = namedtuple('Action', ['name','args'])

class ReplayAgent(QAgent):

    def __init__(self, gamma, lr, action_file, explore_prob, log_file, max_layers):
        self.discount = gamma
        self.lr = lr
        self.log = log_file
        self._load_histories()
        self._set_actions(action_file)
        self.numIters = 0
        self.explore_prob = explore_prob
        self.weights = np.zeros(30)
        self.max_layers = max_layers


    def _load_histories(self):
        self.histories = []
        with open(self.log) as file:
            for line in file:
                self.histories.append(eval(line))

    def _preset_actions(self):
        '''
        Randomly sample a history to provide the sequence of actions to explore
        '''
        self.saved_actions, self.final_reward = random.choice(self.histories)


    def replay_learn(self):
        '''
        Offline learning: reset actions each time function is called, don't log history,
        and return the final reward associated with the saved history
        '''
        state = []
        self._preset_actions()
        self.numIters = 0
        for action in self.saved_actions[:-2]:
            self.numIters += 1
            self.update(state, action)
            state.append(action)
        self.weights += self.final_reward * self.featurize(state, action)
        return self.final_reward
    
