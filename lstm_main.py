'''
main.py

Runs the training of the agent
'''
from LstmReplayAgent import LSTMReplayAgent

import matplotlib.pyplot as plt
import random, csv

def main():
    agent = LSTMReplayAgent(gamma=1, lr=.0001,
                            action_file='actions.json', explore_prob=0.1,
                            log_file='random_history.txt', max_layers=4)
    for i in range(300):
        print(i)
        agent.replay_learn()
    x, y = [], []
    for i in range(100):
        print(i)
        x.append(i)
        y.append(agent.learn())

    with open("lstm_offline.csv", 'w') as file:
        csv_writer = csv.writer(file)
        for entry in y:
            csv_writer.writerow(y)

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
