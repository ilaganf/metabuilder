'''
Agent.py

Defines the Q-learning agent
'''
import random
from collections import namedtuple, defaultdict

import neural
from read_actions import get_actions


Action = namedtuple('Action', ['name', 'args'])

class QAgent:

    def __init__(self, gamma, lr, action_file, exploreProb, logFile):
        self.discount = gamma
        self.lr = lr
        self.state = []
        self._set_actions(action_file)
        self.numIters = 0
        self.exploreProb = exploreProb
        self.weights = np.ones(10)
        self.log = logFile


    def _set_actions(self, file):
        self.actions = [Action(act) for act in get_actions(file)]


    def _successors(self):
        if self._check_end_state():
            return None
        return [act for act in self.actions if self._consider_act(act)]


    def _consider_act(self, action):
        '''
        Prunes valid actions based on prior knowledge
        (Like why on earth would you do two batchnorms in a row?)
        '''
        prev_layer = None if not self.state else self.state[-1]
        if not prev_layer:
            return True

        # Max 2 pooling layers in a row
        if action.name in ['ap','mp']:
            return len(self.state) == 1 or self.state[-2].name not in ['ap','mp']

        # Only dense layers after flatten
        if action.name == 'f' or action.name == 'd':
            return self.state[-1].name in ['f','d']

        # No sequential batchnorm layers
        if action.name == 'b':
            return self.state[-1].name != 'b'

        return True


    def _check_end_state(self):
        return self.state and self.state[-1].name == 'o'


    def get_reward(self):
        if not self._check_end_state():
            return 0
        return neural.eval_action(self.state)


    def get_action(self):
        self.numIters += 1
        if random.random() < self.exploreProb:
            return random.choice(self._successors)
        else:
            return max((self.calcQ(act), act) for act in self._successors)


    def featurize(self, sprime):
        '''
        Override me for different problems
        '''
        features = np.zeros(10)
        for layer in sprime:
            if layer.name == 'c':
                features[0] += 1
                features[1] += layer.args['filters'] * layer.args['kernel_size'][0]**2
            if layer.name == 'mp':
                features[2] += 1
                features[3] += layer.args['pool_size'][0]**2
            if layer.name == 'ap':
                features[4] += 1
                features[5] += layer.args['pool_size'][0]**2
            if layer.name == 'd':
                features[6] += 1
                features[7] += layer.args['units']
            features[8] += layer.name == 'f'
            features[9] += layer.name == 'b'
        return features


    def calcQ(self, act):
        features = self.featurize(self.state + [act])
        return np.dot(features, self.weights)


    def explore(self):
        while True:
            action = self.get_action()
            self.update(action)
            self.state.append(action)
            if self._check_end_state(): break
        history = (self.state, self.get_reward())
        self.record(history)


    def record(self, history):
        with open(self.log, 'a') as file:
            file.write(history)
            file.write('\n')


    def update(self, action):
        if self._check_end_state(): return
        v_opt = max(self.calcQ(act) for act in self._successors)
        q_opt = self.calcQ(action)

        factor = self.lr * (q_opt - (reward + self.discount * v_opt))
        self.weights -= factor * self.weights


    def learn(self):
        self.state = []
        self.explore()
