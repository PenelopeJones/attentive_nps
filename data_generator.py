"""
Script for generating 1-dimensional train and test toy regression data from a noisy function.
"""
import numpy as np

from data_utils import x_generator, noisy_function

min_x = -3.5
max_x = 3.5
n_points = 20
std=3.0

#Generate 5 sets of values of x in the range [min_x, max_x], to be used for training and testing
for i in range(5):
    X_train = x_generator(min_x, max_x, n_points)
    y_train = noisy_function(X_train, std)
    np.save('data/xtrain_1dreg' + str(i) + '.npy', X_train)
    np.save('data/ytrain_1dreg' + str(i) + '.npy', y_train)

    #Generate target values of x and y for sampling from later on
    X_test = np.expand_dims(np.linspace(min_x - 0.5, max_x + 0.5, 140), axis = 1)
    y_test = noisy_function(X_test, 3)

    np.save('data/xtest_1dreg' + str(i) + '.npy', X_test)
    np.save('data/ytest_1dreg' + str(i) + '.npy', y_test)

