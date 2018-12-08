'''
main.py

Runs the training of the agent
'''
from ReplayAgent import ReplayAgent
import matplotlib.pyplot as plt
import csv

def main():
    agent = ReplayAgent(gamma=0.95, lr=.0001,
                   action_file='actions.json', explore_prob=0.01, log_file='random_history.txt', max_layers=4)
    x, y = [], []
    for i in range(300):
        print(i)
        x.append(i)
        agent.replay_learn()
    for i in range(100):
        y.append(agent.learn())

    with open("linear_offline.csv", 'w') as file:
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
