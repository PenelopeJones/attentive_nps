"""
Utility functions for 1d regression experiments
"""
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler


def transform_data(X_train, y_train, X_test, y_test):
    """
    Apply feature scaling to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, x_scaler, y_scaler

def x_generator(min_x, max_x, n_points):
    """
    Generates n_points 1D values of x distributed between min_x and max_x.
    :param min_x: The minimum value of x that can be sampled.
    :param max_x: The maximum value of x that can be sampled
    :param n_points: An integer indicating the number of points that will be sampled.
    :return:
    """
    x1 = min_x + 0.1
    x2 = max_x - 0.1
    x = (x2 - x1) * np.random.random(n_points - 2) + x1
    x = np.insert(x, 0, values = min_x)
    x = np.append(x, values = max_x)
    x = np.reshape(x, (x.shape[0], 1))
    return x

def noisy_function(x, std):
    """
    Generates the noisy function that is used as an example in https://arxiv.org/pdf/1511.03243.pdf.
    :param x: A vector of 1 dimensional inputs.
    :param std: An integer indicating the magnitude of noise.
    :return: y, a vector of outputs y = f(x).
    """
    y = x**3 + np.random.normal(loc=0, scale=std, size=(x.shape[0],1))
    return y


def nlpd(pred_mean_vec, pred_var_vec, targets):
    """
    Computes the negative log predictive density for a set of targets assuming a Gaussian noise model.
    :param pred_mean_vec: predictive mean of the model at the target input locations
    :param pred_var_vec: predictive variance of the model at the target input locations
    :param targets: target values
    :return: nlpd (negative log predictive density)
    """
    assert len(pred_mean_vec) == len(pred_var_vec)  # pred_mean_vec must have been evaluated at xs corresponding to ys.
    assert len(pred_mean_vec) == len(targets)
    nlpd = 0
    index = 0
    n = len(targets)  # number of data points
    pred_mean_vec = np.array(pred_mean_vec).reshape(n, )
    pred_var_vec = np.array(pred_var_vec).reshape(n, )
    pred_std_vec = np.sqrt(pred_var_vec)
    targets = np.array(targets).reshape(n, )
    for target in targets:
        density = scipy.stats.norm(pred_mean_vec[index], pred_std_vec[index]).pdf(target)
        nlpd += -np.log(density)
        index += 1
    nlpd /= n
    return nlpd

