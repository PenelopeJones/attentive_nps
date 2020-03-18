"""
Script for training an Attentive Neural Process on 1d toy data for regression.

In stage 1, we fit a GP to the toy data.

We then sample functions from this GP and use 'context points' generated from these functions to train an ANP.
In this way, the ANP should learn to model the GP prior.

Then at test time we can use our original training data as the 'context points' for our unseen test data.

Based on the work carried out in this paper:
Attentive Neural Processes:
"""
import time
import warnings
import argparse
import sys

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.append('../')

from anp_model import AttentiveNP
from data_utils import transform_data, nlpd


def main(batch_size, learning_rate, iterations, r_size, det_encoder_hidden_size,
         det_encoder_n_hidden, lat_encoder_hidden_size, lat_encoder_n_hidden,
         decoder_hidden_size, decoder_n_hidden, testing, plotting):
    """
    :param batch_size: Integer, describing the number of times we should sample the set
                                of context points used to form the aggregated embedding during
                                training, given the number of context points to be sampled
                                N_context. When testing this is set to 1
    :param learning_rate: A float number, describing the optimiser's learning rate
    :param iterations: An integer, describing the number of iterations. In this case it also
                       corresponds to the number of times we sample the number of context points
                       N_context

    :param r_size: An integer describing the dimensionality of the embedding / context vector r
    :param det_encoder_hidden_size: An integer describing the number of nodes per hidden layer in the
                                deterministic encoder neural network
    :param encoder_n_hidden: An integer describing the number of hidden layers in the encoder neural
                             network
    :param decoder_hidden_size: An integer describing the number of nodes per hidden layer in the
                                decoder neural network
    :param decoder_n_hidden: An integer describing the number of hidden layers in the decoder neural
                             network
    :param testing: A Boolean variable; if true, during testing the RMSE on test and train data '
                             'will be printed after a specific number of iterations.
    :param plotting: A Boolean variable; if true, during testing the context points and predicted mean '
                             'and variance will be plotted after a specific number of iterations.
    :return:
    """
    warnings.filterwarnings('ignore')

    r2_list = []
    rmse_list = []
    mae_list = []
    time_list = []
    print('\nBeginning training loop...')
    j = 0
    for i in range(1,2):
        start_time = time.time()

        #Load training data
        x_train = np.load('data/xtrain_1dreg' + str(i) + '.npy')
        y_train = np.load('data/ytrain_1dreg' + str(i) + '.npy')

        #Generate target values of x and y for sampling from later on
        x_test = np.load('data/xtest_1dreg' + str(i) + '.npy')
        y_test = np.load('data/ytest_1dreg' + str(i) + '.npy')

        #Transform the data: standardise to zero mean and unit variance
        x_train, y_train, x_test, y_test, x_scaler, y_scaler = transform_data(x_train, y_train, x_test,
                                                                    y_test)

        print('... building model.')

        #Build the Attentive Neural Process model, with the following architecture:
        #(x, y)_i --> encoder --> r_i
        #r = average(r_i)
        #(x*, r) --> decoder --> y_mean*, y_var*
        #The encoder and decoder functions are neural networks, with size and number of layers being
        # hyperparameters to be selected.
        anp = AttentiveNP(x_size=x_train.shape[1], y_size=y_train.shape[1], r_size=r_size,
                  det_encoder_hidden_size=det_encoder_hidden_size, det_encoder_n_hidden=det_encoder_n_hidden,
                  lat_encoder_hidden_size=lat_encoder_hidden_size, lat_encoder_n_hidden=lat_encoder_n_hidden,
                  decoder_hidden_size=decoder_hidden_size, decoder_n_hidden=decoder_n_hidden, attention_type="multihead")

        print('... training.')

        #Train the model(NB can replace x_test, y_test with x_valid and y_valid if planning to use
        # a cross validation set)
        anp.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_scaler=x_scaler,
                  y_scaler=y_scaler, batch_size=batch_size, lr=learning_rate,
                  iterations=iterations, testing=testing, plotting=plotting)

        #Testing: the 'context points' when testing are the entire training set, and the 'target
        # points' are the entire test set.
        x_context = torch.tensor(np.expand_dims(x_train, axis=0))
        y_context = torch.tensor(np.expand_dims(y_train, axis=0))
        x_test = torch.tensor(np.expand_dims(x_test, axis=0))

        #Predict mean and error in y given the test inputs x_test
        _, predict_test_mean, predict_test_var = anp.predict(x_context, y_context, x_test)

        predict_test_mean = np.squeeze(predict_test_mean.data.numpy(), axis=0)
        predict_test_var = np.squeeze(predict_test_var.data.numpy(), axis=0)

        # We transform the standardised predicted and actual y values back to the original data
        # space
        y_mean_pred = y_scaler.inverse_transform(predict_test_mean)
        y_var_pred = y_scaler.var_ * predict_test_var
        y_test = y_scaler.inverse_transform(y_test)

        #Calculate relevant metrics
        score = r2_score(y_test, y_mean_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_mean_pred))
        mae = mean_absolute_error(y_test, y_mean_pred)
        nlpd_test = nlpd(y_mean_pred, y_var_pred, y_test)
        time_taken = time.time() - start_time

        np.save('ytest_mean_pred_1dreg' + str(i) + '_anp.npy', y_mean_pred)
        np.save('ytest_var_pred_1dreg' + str(i) + '_anp.npy', y_var_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))
        print("NLPD: {:.4f}".format(nlpd_test))
        print("Execution time: {:.3f}".format(time_taken))
        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        time_list.append(time_taken)

        j += 1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    time_list = np.array(time_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list),
                                                np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list),
                                               np.std(rmse_list) / np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list),
                                                np.std(mae_list) / np.sqrt(len(mae_list))))
    print("mean Execution time: {:.3f} +- {:.3f}\n".format(np.mean(time_list),
                                                           np.std(time_list)/
                                                           np.sqrt(len(time_list))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The number of samples to take of the context set, given the number of'
                             ' context points that should be selected.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The training learning rate.')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='Number of training iterations.')
    parser.add_argument('--r_size', type=int, default=2,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('--det_encoder_hidden_size', type=int, default=4,
                        help='Dimensionality of deterministic encoder hidden layers.')
    parser.add_argument('--det_encoder_n_hidden', type=int, default=2,
                        help='Number of deterministic encoder hidden layers.')
    parser.add_argument('--lat_encoder_hidden_size', type=int, default=4,
                        help='Dimensionality of latent encoder hidden layers.')
    parser.add_argument('--lat_encoder_n_hidden', type=int, default=2,
                        help='Number of latent encoder hidden layers.')
    parser.add_argument('--decoder_hidden_size', type=int, default=4,
                        help='Dimensionality of decoder hidden layers.')
    parser.add_argument('--decoder_n_hidden', type=int, default=2,
                        help='Number of decoder hidden layers.')
    parser.add_argument('--testing', default=False,
                        help='If true, during testing the RMSE on test and train data '
                             'will be printed after specific numbers of iterations.')
    parser.add_argument('--plotting', default=False,
                        help='If true, at the end of training a plot will be produced.')
    args = parser.parse_args()

    main(args.batch_size, args.learning_rate,
         args.iterations, args.r_size, args.det_encoder_hidden_size, args.det_encoder_n_hidden,
         args.lat_encoder_hidden_size, args.lat_encoder_n_hidden,
         args.decoder_hidden_size, args.decoder_n_hidden, args.testing, args.plotting)