'''
main.py

Runs the training of the agent
'''
from LstmReplayAgent import LSTMReplayAgent

import matplotlib.pyplot as plt

def main():
    agent = ReplayAgent(gamma=1, lr=.0001,
                        action_file='actions.json', exploreProb=0.1, log_file='history.txt')
    x, y = [], []
    for i in range(100):
        print(i)
        agent.replay_learn()
    for i in range(30):
        x.append(i)
        y.append(agent.learn())

    plt.figure()
    plt.plot(x, y)
    plt.savefig("ok.png")
    # conv (C): filters, kernel_size, strides, padding='SAME', activation=tf.nn.relu

    # batchnorm (B)

    # max_pool (MP): pool_size, strides, padding='SAME'

    # avg_pool (AP): pool_size, strides, padding='SAME'

    # dense (D): units

if __name__ == '__main__':
    main()
