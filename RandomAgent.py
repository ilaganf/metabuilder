from Agent import QAgent
import random, time
from collections import namedtuple

Action = namedtuple('Action', ['name', 'args'])

class RandomAgent(QAgent):

    def get_action(self, state):
        if len(state) == self.max_layers - 1 and all([prev_state.name != 'f' for prev_state in state]):
            return Action(name='f', args={})
        if len(state) == self.max_layers:
            return Action(name='o', args={'units':10})

        return random.choice(self._successors(state))

    def learn(self):
        state = []
        while True:
            action = self.get_action(state)
            if action.name=='lr':
                reward = self.get_reward(state, action)
                break
            state.append(action)
        history = (state, reward)
        self.record(history)

if __name__=="__main__":
    for n_layers in range(2, 11):
        agent = RandomAgent(gamma=1, lr=0.001, action_file="actions.json",
                            explore_prob=1.1, log_file='random_history.txt',
                            max_layers=n_layers)

        for i in range(100):
            agent.learn()
