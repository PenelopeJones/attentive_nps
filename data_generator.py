"""
Script for generating 1-dimensional train and test toy regression data from a noisy function.
"""
import numpy as np
from gp_sampler import GPDataGenerator
import matplotlib.pyplot as plt

import pdb

data_generator = GPDataGenerator()
dataname = "1DGPSquaredExponentialb_"
x_trains = []
y_trains = []
x_tests = []
y_tests = []
#Generate 5 sets of values of x in the range [min_x, max_x], to be used for training and testing
plt.figure()
for i in range(5):
    n_train = np.random.randint(low = 400, high = 900)
    n_test = int(0.15*n_train)
    print(n_train)
    print(n_test)
    x_train, y_train, x_test, y_test = data_generator.sample(train_size=n_train, test_size=n_test, x_min=-3, x_max=3)

    x_trains.append(x_train)
    y_trains.append(y_train)
    x_tests.append(x_test)
    y_tests.append(y_test)

    if i ==0:
        plt.scatter(x_train, y_train, c='r', s=2, label="train")
        plt.scatter(x_test, y_test, c="b", s=2, label="test")
    else:
        plt.scatter(x_train, y_train, c='r', s=2)
        plt.scatter(x_test, y_test, c="b", s=2)
plt.legend()
plt.xlabel("x")
plt.xticks([])
plt.ylabel('f(x)')
plt.yticks([])
plt.show()
plt.savefig(dataname + "plot.png")

x_trains = np.array(x_trains)
y_trains = np.array(y_trains)
x_tests = np.array(x_tests)
y_tests = np.array(y_tests)

pdb.set_trace()


ptr = "data/"
np.save(ptr + dataname + "X_trains.npy", x_trains)
np.save(ptr + dataname + "y_trains.npy", y_trains)
np.save(ptr + dataname + "X_tests.npy", x_tests)
np.save(ptr + dataname + "y_tests.npy", y_tests)

