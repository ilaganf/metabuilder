'''
Agent.py

Defines the Q-learning agent
'''

class QAgent:

    def __init__(self, gamma, lr, action_file, output_layer):
        self.discount = gamma
        self.lr = lr
        self.state = []
        self._set_actions(action_file)
        self.end = output_layer


    def _set_actions(self, file):
        self.actions = None
        pass


    def _successors(self):
        if self._check_end_state():
            return None

        succ_layers = []
        for action in self.actions:
            if self._consider_act(action):
                succ_layers.append(action)
            if prev_layer and prev_layer[-1][0] == 'd':
                if action[0] == 'd':
                    succ_layers.append(action)
            else:
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
        if action[0] in ['ap','mp']:
            pass

        # Only dense layers after flatten
        if action[0] == 'f':
            pass

        # No sequential batchnorm layers
        if action[0] == 'b':
            pass

    def _check_end_state(self):
        return self.state and self.state[-1] == self.end


    def learn(self):
        pass
