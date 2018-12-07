'''
Agent.py

Defines the Q-learning agent
'''
import random
from collections import namedtuple, defaultdict

import neural
import numpy as np
from create_actions import *
from copy import deepcopy
from read_actions import get_actions


max_layers = 10
Action = namedtuple('Action', ['name', 'args'])

class QAgent:

    def __init__(self, gamma, lr, action_file, explore_prob, log_file):
        self.discount = gamma
        self.lr = lr
        self._set_actions(action_file)
        self.numIters = 0
        self.exploreProb = explore_prob
        self.weights = np.zeros(29)
        self.log = log_file


    def _set_actions(self, file):
        self.actions = [Action(name.lower(), args) for name, args in get_actions(file)]


    def _successors(self, state):
        if self._check_end_state(state):
            return None
        return [deepcopy(act) for act in self.actions if self._consider_act(state, act)]


    def _consider_act(self, state, action):
        '''
        Prunes valid actions based on prior knowledge
        (Like why on earth would you do two batchnorms in a row?)
        '''
        prev_layer = None if not state else state[-1]
        if not prev_layer:
            return action.name == 'c'

        if prev_layer.name == 'o':
            return action.name == 'lr'

        if action.name == 'lr':
            return False

        # Max 2 pooling layers in a row
        if action.name in ['ap','mp']:
            return len(state) == 1 or state[-2].name not in ['ap','mp']

        # Only dense layers after flatten
        if action.name == 'f' or action.name == 'd' or action.name == 'o':
            return state[-1].name in ['f','d']

        # No sequential batchnorm layers
        if action.name == 'b':
            return state[-1].name != 'b'

        return True


    def _check_end_state(self, state):
        return state and state[-1].name == 'lr'


    def get_reward(self, state, action):
        if not action.name == 'lr':
            return 0
        return neural.eval_action(state + [action])


    def get_action(self, state):
        '''
        Search action using epsilon greedy approach
        '''
        self.numIters += 1
        if len(state) == max_layers - 1 and all([prev_state.name != 'f' for prev_state in state]):
            return Action(name='f', args={})
        if len(state) == max_layers:
            return Action(name='o', args={'units':10})
            
        if random.random() < self.explore_prob/self.numIters:
            return random.choice(self._successors(state))
        else:
            return max([(self.calcQ(state, act),act) for act \
                        in self._successors(state)], key=lambda x:x[0])[1]


    def featurize(self, state, action):
        '''
        Takes the given state sprime and featurizes it by counting
        how many of each layer type there are and by counting the number
        of parameters (may need more sophisticated features)
        '''
        features = np.zeros(29)
        for layer in state:
            if layer.name == 'c':
                features[0] += 1
                features[1] += layer.args['filters']
                features[2] += layer.args['kernel_size']
                features[3] += layer.args['strides']
            if layer.name == 'mp':
                features[4] += 1
                features[5] += layer.args['pool_size']
                features[6] += layer.args['strides']
            if layer.name == 'ap':
                features[7] += 1
                features[8] += layer.args['pool_size']
                features[9] += layer.args['strides']
            if layer.name == 'd':
                features[10] += 1
                features[11] += layer.args['units']
            features[12] += layer.name == 'f'
            features[13] += layer.name == 'b'
        features[14] = len(state)
        if action.name == 'c':
            features[15] += 1
            features[16] += action.args['filters']
            features[17] += action.args['kernel_size']
            features[18] += action.args['strides']
        if action.name == 'mp':
            features[19] += 1
            features[20] += action.args['pool_size']
            features[21] += action.args['strides']
        if action.name == 'ap':
            features[22] += 1
            features[23] += action.args['pool_size']
            features[24] += action.args['strides']
        if action.name == 'd':
            features[25] += 1
            features[26] += action.args['units']
        features[27] += action.name == 'f'
        features[28] += action.name == 'b'
        #features /= np.linalg.norm(features)
        return features


    def calcQ(self, state, act):
        features = self.featurize(state, act)
        return np.dot(features, self.weights)


    def calcVOpt(self, state, act):
        next_state = state + [act]
        candidates = []
        successors = self._successors(next_state)

        # Current action is taking us into end state
        if successors is None:
            return 0

        for next_action in successors:
            candidates.append(self.calcQ(next_state, next_action))
        return max(candidates)


    def record(self, history):
        '''
        Log a network architecture compiled by the agent
        '''
        with open(self.log, 'a') as file:
            file.write(str(history))
            file.write('\n')


    def update(self, state, action):
        '''
        Use Bellman to update the weights of the model given an action 

        state is stored as part of the object and transitions are 
        deterministic, so only need action to calculate
        '''
        v_opt = self.calcVOpt(state, action)
        q_opt = self.calcQ(state, action)
        r = self.get_reward(state, action)
        factor = self.lr * (r + self.discount * v_opt - q_opt)
        self.weights += factor * self.featurize(state, action)
        return r


    def learn(self):
        state = []
        self.numIters = 0
        while True:
            action = self.get_action(state)
            reward = self.update(state, action)
            state.append(action)
            if self._check_end_state(state): break
        history = (state, reward)
        self.record(history)
        return reward
