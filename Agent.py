'''
Agent.py

Defines the Q-learning agent
'''

class QAgent:

    def __init__(self, gamma, lr, action_file):
        self.discount = gamma
        self.lr = lr
        self.state = []
        # state = [('C', 32, 5, 2), ('P', 3, 2)]
        self._set_actions(action_file)


    def _set_actions(self, file):
        pass


    def learn(self):
        pass
