# ReplayAgent.py
from collections import namedtuple

import numpy as np

from Agent import QAgent

Action = namedtuple('Action', ['name','args'])

class ReplayAgent(QAgent):

    def __init__(self, gamma, lr, action_file, exploreProb, log_file):
        self.discount = gamma
        self.lr = lr
        self.log = log_file
        self._load_histories()
        self._set_actions(action_file)
        self.numIters = 0
        self.exploreProb = exploreProb
        self.weights = np.zeros(29)


    def _load_histories(self):
        self.histories = []
        with open(self.log) as file:
            for line in file:
                self.histories.append(eval(line))


    def _preset_actions(self):
        '''
        Randomly sample a history to provide the sequence of actions to explore
        '''
        self.saved_actions, self.final_reward = np.random.choice(self.histories)


    def learn(self):
        '''
        Main differences: reset actions each time function is called, don't log history,
        and return the final reward associated with the saved history
        '''
        state = []
        self._preset_actions()
        self.numIters = 0
        for action in self.saved_actions:
            self.numIters += 1
            self.update(state, action)
            state.append(action)
        return self.final_reward
    
