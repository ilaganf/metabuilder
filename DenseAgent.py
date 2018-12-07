# DenseAgent.py

from Agent import QAgent

import random

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

import neural
from Agent import QAgent

DIM = 29

class DenseAgent(QAgent):
    def __init__(self, gamma, lr, action_file, exploreProb, logFile):
        self.discount = gamma
        self.lr = lr
        self._set_actions(action_file)
        self.numIters = 0
        self.exploreProb = exploreProb
        self.log = logFile
        self.model = self.compile_model()

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
        features = self.featurize(state, act)[None, :]
        return self.model.predict(features)


    def update(self, state, action):
        v_opt = self.calcVOpt(state, action)
        r = np.array([self.get_reward(state, action)])
        y = np.array(r + self.discount * v_opt)
        print(y, self.calcQ(state, action))
        self.model.fit(self.featurize(state, action)[None,:], y)
        return r