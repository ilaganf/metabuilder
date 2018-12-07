
from LstmAgent import LSTMAgent
import numpy as np
import random

class LSTMSarsa(LSTMAgent):

    def update(self, s, a):
        qnext = self.calcQ(sp, ap)
        r = np.array([self.get_reward(s, a)])
        y = np.array(r + self.discount * qnext)
        print(y, self.calcQ(s, a))
        self.model.fit(self.featurize(s, a)[np.newaxis, :, :], y)
        return r


    #def replay_learn(self, numiter=1000):
    #    '''
    #    Offline learning: reset actions each time function is called, don't log history,
    #    and return the final reward associated with the saved history
    #    '''
    #    self._preset_actions()
    #    self.numIters = 0
    #    for i in range(numiter):
    #        self.numIters += 1
    #        s, sp, ap = self.sample_sarsa()
    #        self.sarsa_update(s, sp[-1], sp, ap)
    #    return 0
