# train.py

import numpy as np

from Agent import QAgent

class ReplayAgent(QAgent):

    def __init__(self, gamma, lr, action_file, exploreProb, log_file):
        self.discount = gamma
        self.lr = lr
        self.state = []
        self._set_actions(log_file)
        self.numIters = 0
        self.exploreProb = exploreProb
        self.weights = np.zeros(10)
        self.log = logFile


    def _set_actions(self, file):
        self.actions = [Action(name.lower(), args) for name, args in get_actions(file)]


    def learn(self):
        '''
        Main differences: reset actions, don't log history
        '''
        self.state = []
        self._set_actions(self.log)
        while True:
            action = self.get_action()
            reward = self.update(action)
            self.state.append(action)
            if self._check_end_state(): break
        print(self.weights)
        return reward
    