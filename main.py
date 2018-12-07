'''
main.py

Runs the training of the agent
'''
from ReplayAgent import ReplayAgent
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt


def main():
    agent = ReplayAgent(gamma=0.95, lr=.0001,
                   action_file='actions.json', exploreProb=0.1, log_file='random_history.txt',
                   max_layers=7)
    x, y = [], []
    for i in range(1, 501):
        print("Experience replay iteration ", i)
        agent.replay_learn()

    explore = 1
    agent.explore_prob = explore
    for i in range(1, 101):
        if i % 10 == 0:
            explore -= .1
            agent.explore_prob = explore
        x.append(i)
        y.append(agent.learn())
        # y.append(agent.replay_learn())
    # x.append(99)
    # y.append(agent.learn())

    plt.figure()
    plt.plot(x, y)
    plt.savefig("linear_performance.png")
    # conv (C): filters, kernel_size, strides, padding='SAME', activation=tf.nn.relu

    # batchnorm (B)

    # max_pool (MP): pool_size, strides, padding='SAME'

    # avg_pool (AP): pool_size, strides, padding='SAME'

    # dense (D): units

if __name__ == '__main__':
    main()
