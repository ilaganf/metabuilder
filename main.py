'''
main.py

Runs the training of the agent
'''
from Agent import QAgent

def main():
    agent = QAgent(gamma=1, lr=.95,
                   action_file='actions.json', exploreProb=.01, logFile='history.txt')

    for _ in range(100):
        agent.learn()
    # conv (C): filters, kernel_size, strides, padding='SAME', activation=tf.nn.relu

    # batchnorm (B)

    # max_pool (MP): pool_size, strides, padding='SAME'

    # avg_pool (AP): pool_size, strides, padding='SAME'

    # dense (D): units

if __name__ == '__main__':
    main()
