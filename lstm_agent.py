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

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

max_layers = 5
DIM = 16
Action = namedtuple('Action', ['name', 'args'])

class QAgent:

    def __init__(self, gamma, lr, action_file, exploreProb, logFile):
        self.discount = gamma
        self.lr = lr
        self._set_actions(action_file)
        self.numIters = 0
        self.exploreProb = exploreProb
        self.weights = np.zeros(29)
        self.log = logFile
        self.model = self.compile_model()

    def _set_actions(self, file):
        self.actions = [Action(*act) for act in get_actions(file)]


    def _successors(self, state):
        if self._check_end_state(state):
            return None
        return [deepcopy(act) for act in self.actions if self._consider_act(state, act)]

    def compile_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(None, DIM), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
        return model

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

        # Only dense layers after flatten
        if prev_layer.name in ['f', 'd']:
            return action.name == 'd' or action.name == 'o'

        # Max 2 pooling layers in a row
        if action.name in ['ap','mp']:
            return len(state) == 1 and state[-1].name not in ['ap','mp']

        if action.name == 'd' or action.name == 'o':
            return prev_layer.name in ['f', 'd']

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
        self.numIters += 1
        if len(state) == max_layers - 1 and all([prev_state.name != 'f' for prev_state in state]):
            return Action(name='f', args={})
        if len(state) == max_layers:
            return Action(name='o', args={'units':10})
            
        if random.random() < self.exploreProb:
            return random.choice(self._successors(state))
        else:
            return max([(self.calcQ(state, act),act) for act in self._successors(state)], key=lambda x:x[0])[1]


    def featurize(self, state, action):
        '''
        Override me for different problems
        '''
        features = np.zeros((len(state) + 1, 16))
        for i, layer in enumerate(state + [action]):
            if layer.name == 'c':
                features[i, 0] += 1
                features[i, 1] += layer.args['filters']
                features[i, 2] += layer.args['kernel_size']
                features[i, 3] += layer.args['strides']
            if layer.name == 'mp':
                features[i, 4] += 1
                features[i, 5] += layer.args['pool_size']
                features[i, 6] += layer.args['strides']
            if layer.name == 'ap':
                features[i, 7] += 1
                features[i, 8] += layer.args['pool_size']
                features[i, 9] += layer.args['strides']
            if layer.name == 'd':
                features[i, 10] += 1
                features[i, 11] += layer.args['units']
            if layer.name == 'lr':
                features[i, 12] += layer.name == 'lr'
                features[i, 13] += layer.args['lr']

            features[i, 14] += layer.name == 'f'
            features[i, 15] += layer.name == 'b'
        return features


    def calcQ(self, state, act):
        features = self.featurize(state, act)[np.newaxis, :, :]
        return self.model.predict(features)

    def calcVOpt(self, state, act):
        next_state = state + [act]
        candidates = [0.1]
        successors = self._successors(next_state)

        # Current action is taking us into end state
        if successors is None:
            return 0

        for next_action in successors:
            candidates.append(self.calcQ(next_state, next_action))
        return max(candidates)

    def explore(self):
        state = []
        while True:
            action = self.get_action(state)
            reward = self.update(state, action)
            state.append(action)
            if self._check_end_state(state): break
        print(self.weights)
        history = (state, reward[0])
        self.record(history)
        return(reward)

    def record(self, history):
        with open(self.log, 'a') as file:
            file.write(str(history))
            file.write('\n')


    def update(self, state, action):
        #if self._check_end_state():
        #    q_opt = self.calcQ(action)
        #    reward = self.get_reward()
        #    factor = self.lr * (q_opt - reward)
        #    self.weights += factor * self.featurize(self.state)
        #    print(factor, self.weights, reward)
        #    return reward

        v_opt = self.calcVOpt(state, action)
        #q_opt = self.calcQ(state, action)
        r = np.array([self.get_reward(state, action)])
        y = np.array(r + self.discount * v_opt)
        print(y, self.calcQ(state, action))
        self.model.fit(self.featurize(state, action)[np.newaxis, :, :], y)
        #self.weights += factor * self.featurize(state, action)
        return r

    def learn(self):
        self.state = []
        return self.explore()
