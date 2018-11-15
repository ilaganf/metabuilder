'''
Agent.py

Defines the Q-learning agent
'''
from collections import namedtuple

from read_actions import get_actions


Action = namedtuple('Action', ['name', 'args'])

class QAgent:

    def __init__(self, gamma, lr, action_file, output_layer):
        self.discount = gamma
        self.lr = lr
        self.state = []
        self._set_actions(action_file)
        self.end = output_layer


    def _set_actions(self, file):
        self.actions = [Action(act) for act in get_actions(file)]


    def _successors(self):
        if self._check_end_state():
            return None

        succ_layers = []
        for action in self.actions:
            if self._consider_act(action):
                succ_layers.append(action)

        return succ_layers


    def _consider_act(self, action):
        '''
        Prunes valid actions based on prior knowledge
        '''
        prev_layer = None if not self.state else self.state[-1]
        if not prev_layer:
            return True

        # Max 2 pooling layers in a row
        if action.name in ['ap','mp']:
            return len(self.state) <= 1 or self.state[-2].name not in ['ap','mp']

        # Only dense layers after flatten
        if action.name == 'f' or action.name == 'd':
            return self.state[-1].name in ['f','d']

        # No sequential batchnorm layers
        if action.name == 'b':
            if len(self.state) > 1 and self.state[-1].name == 'b':
                return False

        return True

    def _check_end_state(self):
        return self.state and self.state[-1] == self.end


    def learn(self):
        pass
