'''
Agent.py

Defines the Q-learning agent
'''
import random, csv
from collections import namedtuple, defaultdict
from copy import deepcopy

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

import neural
from Agent import QAgent
from create_actions import *
from read_actions import get_actions

DIM = 16
Action = namedtuple('Action', ['name', 'args'])

class LSTMAgent(QAgent):

    def __init__(self, gamma, lr, action_file, explore_prob, log_file, max_layers):
        self.discount = gamma
        self.lr = lr
        self._set_actions(action_file)
        self.numIters = 0
        self.explore_prob = explore_prob
        self.log = log_file
        self.model = self.compile_model()
        self.max_layers = max_layers

    def compile_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(None, DIM)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
        return model


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


    def update(self, state, action, final_only=True):
        v_opt = self.calcVOpt(state, action)
        r = np.array([self.get_reward(state, action)])
        y = np.array(r + self.discount * v_opt)
        print(y, self.calcQ(state, action))
        if final_only:
            if np.sum(r) > 0:
                self.model.fit(self.featurize(state, action)[np.newaxis, :, :], y)
        else:
            self.model.fit(self.featurize(state, action)[np.newaxis, :, :], y)
        return r

if __name__=='__main__':
    agent = LSTMAgent(gamma=0.95, lr=0.0001, action_file='actions.json',
                      explore_prob=0.1, log_file='history.txt', max_layers=3)
    x, y = [], []
    for i in range(100):
        output = agent.learn()[0]
        print(output)
        y.append(output)
    with open("lstm_online.csv", 'w') as file:
        csv_writer = csv.writer(file)
        for entry in y:
            csv_writer.writerow(y)
