
from LstmReplayAgent import LSTMReplayAgent

class LSTMSarsa(LSTMReplayAgent):

    def sarsa_update(self, s, a, sp, ap):
        qnext = self.calcQ(sp, ap)
        r = np.array([self.get_reward(s, a)])
        y = np.array(r + self.discount * qnext)
        print(y, self.calcQ(s, a))
        self.model.fit(self.featurize(s, a)[np.newaxis, :, :], y)
        return r


    def sample_sarsa(self):
        hist = random.sample(self.history)
        s_ind = random.randint(0, len(hist)-2)
        return hist[:s_ind], hist[:s_ind+1], hist[s_ind+1]


    def replay_learn(self, numiter=1000):
        '''
        Offline learning: reset actions each time function is called, don't log history,
        and return the final reward associated with the saved history
        '''
        self._preset_actions()
        self.numIters = 0
        for i in range(numiter):
            self.numIters += 1
            s, sp, ap = self.sample_sarsa()
            self.sarsa_update(s, sp[-1], sp, ap)
        return 0