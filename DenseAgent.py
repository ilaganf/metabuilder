# DenseAgent.py

from Agent import QAgent

import random

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

import neural, csv
from Agent import QAgent

DIM = 30

class DenseAgent(QAgent):
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
        model.add(Dense(128, activation='relu', input_shape=(None, DIM)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
        return model

    def calcQ(self, state, act):
        features = self.featurize(state, act)[np.newaxis, np.newaxis, :]
        return self.model.predict(features)


    def update(self, state, action):
        v_opt = self.calcVOpt(state, action)
        r = np.array([[[self.get_reward(state, action)]]])
        y = np.array(r + self.discount * v_opt)
        print(y, self.calcQ(state, action))
        self.model.fit(self.featurize(state, action)[np.newaxis, np.newaxis, :], y)
        return r

if __name__=='__main__':
    agent = DenseAgent(gamma=0.95, lr=0.0001, action_file='actions.json',
                      explore_prob=0.1, log_file='history.txt', max_layers=3)
    x, y = [], []
    for i in range(100):
        output = agent.learn()[0]
        print(output)
        y.append(output)
    with open("mlp_online.csv", 'w') as file:
        csv_writer = csv.writer(file)
        for entry in y:
            csv_writer.writerow(y)
