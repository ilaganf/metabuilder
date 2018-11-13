import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test

def model_init(layers):
    model = None
    ############################################################################
    # TODO: Construct a model that performs well on CIFAR-10                   #
    ############################################################################
    input_shape=(32,32,3)
    initializer = tf.variance_scaling_initializer(scale=2.0)
    channel_1, channel_2 = 64, 32
    pool_size_1, pool_size_2 = (3,3), (3,3)
    hidden_size = 200
    reg = 0.001
    num_classes = 10

    #layers = [
    #    tf.layers.Conv2D(input_shape=input_shape, filters=channel_1, kernel_size=(5,5), strides=3, padding="same", activation=tf.nn.relu, kernel_initializer=initializer, kernel_regularizer=tf.contrib.layers.l2_regularizer(reg)),
    #    tf.layers.BatchNormalization(),
    #    tf.layers.MaxPooling2D(pool_size=pool_size_1, strides=2, padding="SAME"),
    #    tf.layers.Conv2D(filters=channel_2, kernel_size=(5,5), strides=1, padding="same", activation=tf.nn.relu, kernel_initializer=initializer, kernel_regularizer=tf.contrib.layers.l2_regularizer(reg)),
    #    tf.layers.BatchNormalization(),
    #    tf.layers.MaxPooling2D(pool_size=pool_size_2, strides=2, padding="SAME"),
        #tf.layers.Conv2D(filters=channel_3, kernel_size=(3,3), strides=1, padding="same", activation=tf.nn.relu, kernel_initializer=initializer, kernel_regularizer=tf.contrib.layers.l2_regularizer(reg)),
        #tf.layers.MaxPooling2D(pool_size=pool_size_2, strides=1),
        #tf.layers.BatchNormalization(),
    #    tf.layers.Flatten(),
    #    tf.layers.Dense(384, kernel_initializer=initializer, kernel_regularizer=tf.contrib.layers.l2_regularizer(reg), activation=tf.nn.relu),
    #    tf.layers.Dense(192, kernel_initializer=initializer, kernel_regularizer=tf.contrib.layers.l2_regularizer(reg), activation=tf.nn.relu),
    #    tf.layers.Dense(num_classes, kernel_initializer=initializer, kernel_regularizer=tf.contrib.layers.l2_regularizer(reg), activation=tf.nn.softmax)
    #]

    model = tf.keras.Sequential(layers)
    return model

def create_layers(actions):
    layers = []
    input_shape = (32, 32, 3)
    for i, (action, value) in enumerate(actions):
        if i == 0:
            value['input_shape'] = input_shape
            layers += [tf.layers.Conv2D(**value)]
        if action == 'c':
            layers += [tf.layers.Conv2D(**value)]
        if action == 'b':
            layers += [tf.layers.BatchNormalization()]
        if action == 'mp':
            layers += [tf.layers.MaxPooling2D(**value)]
        if action == 'ap':
            layers += [tf.layers.AveragePooling2D(**value)]
        if action == 'f':
            layers += [tf.layers.Flatten()]
        if action == 'd':
            layers += [tf.layers.Dense(**value)]
    return layers

def optimizer_init_fn():
    return tf.train.AdamOptimizer()

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    device = '/gpu:0'
    print_every = 700
    num_epochs = 10
    actions =   [('c', {'filters':64, 'kernel_size':(5,5), 'strides':3, 'padding':'SAME'}),
                 ('b', {}),
                 ('mp', {'pool_size':(3,3), 'strides':2, 'padding':'SAME'}),
                 ('c', {'filters':32, 'kernel_size':(5,5), 'strides':33, 'padding':'SAME'}),
                 ('mp', {'pool_size':(3,3), 'strides':2, 'padding':'SAME'}),
                 ('f', {}),
                 ('d', {'units':10})]
    layers = create_layers(actions)
    model=model_init(layers)

    learning_rate = 0.001
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=X_train,
              y=to_categorical(y_train, num_classes=10),
              batch_size=None,
              epochs=num_epochs,
              verbose=1,
              validation_data=(X_val, to_categorical(y_val, num_classes=10)))

if __name__=='__main__':
    main()
