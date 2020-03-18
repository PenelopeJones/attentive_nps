"""
Conditional Neural Process (CNP): CNPs bridge the gap between neural networks and Gaussian
processes, allowing a distribution over functions to be learned and enabling the uncertainty
in a prediction to be estimated. They scale as O(n + m) where n is the number of training
points and m is the number of test points, in contrast to exact GPs which scale as O((n+m)^3).

Based on the work carried out in this paper:
Conditional Neural Processes: Garnelo M, Rosenbaum D, Maddison CJ, Ramalho T, Saxton D,
Shanahan M, Teh YW, Rezende DJ, Eslami SM. Conditional Neural Processes. In International
Conference on Machine Learning 2018.

"""

import numpy as np
import torch
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from deterministic_encoder import DeterministicEncoder
from latent_encoder import LatentEncoder
from decoder import Decoder
from gp_sampler import GPSampler

from data_utils import nlpd

import pdb

class AttentiveNP():
    """
    The Attentive Neural Process model.
    """
    def __init__(self, x_size, y_size, r_size, det_encoder_hidden_size, det_encoder_n_hidden, lat_encoder_hidden_size,
                 lat_encoder_n_hidden, decoder_hidden_size, decoder_n_hidden, attention_type):
        """

        :param x_size: An integer describing the dimensionality of the input x
        :param y_size: An integer describing the dimensionality of the target variable y
        :param r_size: An integer describing the dimensionality of the embedding / context
                       vector r
        :param det_encoder_hidden_size: An integer describing the number of nodes per hidden
                                    layer in the deterministic encoder NN
        :param det_encoder_n_hidden: An integer describing the number of hidden layers in the
                                 deterministic encoder neural network
        :param lat_encoder_hidden_size: An integer describing the number of nodes per hidden
                                    layer in the latent encoder neural NN
        :param lat_encoder_n_hidden: An integer describing the number of hidden layers in the
                                 latent encoder neural network
        :param decoder_hidden_size: An integer describing the number of nodes per hidden
                                    layer in the decoder neural network
        :param decoder_n_hidden: An integer describing the number of hidden layers in the
                                 decoder neural network
        :param attention_type: The type of attention to be used. A string, either "multihead",
                                "laplace", "uniform", "dot_product"
        """

        self.x_size = x_size
        self.y_size = y_size
        self.r_size = r_size
        self.det_encoder = DeterministicEncoder(x_size, y_size, r_size, det_encoder_n_hidden,
                               det_encoder_hidden_size, attention_type)
        self.lat_encoder = LatentEncoder((x_size + y_size), r_size, lat_encoder_n_hidden,
                                                lat_encoder_hidden_size)
        self.decoder = Decoder((x_size + r_size + r_size), y_size, decoder_n_hidden,
                               decoder_hidden_size)
        self.optimiser = optim.Adam(list(self.det_encoder.parameters()) + list(self.lat_encoder.parameters()) +
                                    list(self.decoder.parameters()))


    def train(self, x_train, y_train, x_test, y_test, x_scaler, y_scaler, batch_size, lr,
              iterations, testing, plotting):
        """
        :param x_train: A tensor with dimensions [N_train, x_size] containing the training
                        data (x values)
        :param y_train: A tensor with dimensions [N_train, y_size] containing the training
                        data (y values)
        :param x_test: A tensor with dimensions [N_test, x_size] containing the test data
                       (x values)
        :param y_test: A tensor with dimensions [N_test, y_size] containing the test data
                       (y values)
        :param x_scaler: The standard scaler used when testing == True to convert the
                         x values back to the correct scale.
        :param y_scaler: The standard scaler used when testing == True to convert the predicted
                         y values back to the correct scale.
        :param batch_size: An integer describing the number of times we should
                                    sample the set of context points used to form the
                                    aggregated embedding during training, given the number
                                    of context points to be sampled N_context. When testing
                                    this is set to 1
        :param lr: A float number, describing the optimiser's learning rate
        :param iterations: An integer, describing the number of iterations. In this case it
                           also corresponds to the number of times we sample the number of
                           context points N_context
        :param testing: A Boolean object; if set to be True, then every 30 iterations the
                        R^2 score and RMSE values will be calculated and printed for
                        both the train and test data
        :return:
        """
        self.gp_sampler = GPSampler(data=(x_train, y_train))
        self.batch_size = batch_size
        self._max_num_context = x_train.shape[0]
        self.iterations = iterations

        #Convert the data for use in PyTorch.
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()

        # At prediction time the context points comprise the entire training set.
        x_tot_context = torch.unsqueeze(x_train, dim=0)
        y_tot_context = torch.unsqueeze(y_train, dim=0)

        for iteration in range(iterations):
            self.optimiser.zero_grad()

            # Randomly select the number of context points N_context (uniformly from 3 to
            # N_train)
            num_context = np.random.randint(low=1, high=self._max_num_context)

            # Randomly select N_context context points from the training data, a total of
            # batch_size times.
            x_context, y_context, x_target, y_target = self.gp_sampler.sample(batch_size=self.batch_size, train_size=50,
                                                           num_context=num_context, x_min=-4, x_max=4)

            x_context = torch.from_numpy(x_context).float()
            y_context = torch.from_numpy(y_context).float()
            x_target = torch.from_numpy(x_target).float()
            y_target = torch.from_numpy(y_target).float()

            # The input to both the deterministic and latent encoder is (x, y)_i for all data points in the set of context
            # points.
            input_context = torch.cat((x_context, y_context), dim=2)
            input_target = torch.cat((x_target, y_target), dim=2)

            #The deterministic encoder outputs the deterministic embedding r.
            r = self.det_encoder.forward(x_context, y_context, x_target)  #[batch_size, N_target, r_size]

            # The latent encoder outputs a prior distribution over the latent embedding z (conditioned only on the context points).
            z_priors, mu_prior, sigma_prior = self.lat_encoder.forward(input_context.float())

            if y_target is not None:
                z_posteriors, mu_posterior, sigma_posterior = self.lat_encoder.forward(input_target.float())
                zs= [dist.sample() for dist in z_posteriors]      #[batch_size, r_size]

            else:
                zs = [dist.sample() for dist in z_priors]      #[batch_size, r_size]

            z = torch.cat(zs)
            z = z.view(-1, self.r_size)


            # The input to the decoder is the concatenation of the target x values, the deterministic embedding r and the latent variable z
            # the output is the predicted target y for each value of x.
            dists_y, _, _ = self.decoder.forward(x_target.float(), r.float(), z.float())

            # Calculate the loss
            log_ps = [dist.log_prob(y_target[i, ...].float()) for i, dist in enumerate(dists_y)]
            log_ps = torch.cat(log_ps)

            kl_div = [kl_divergence(z_posterior, z_prior).float() for z_posterior, z_prior in zip(z_posteriors, z_priors)]
            kl_div = torch.tensor(kl_div)

            loss = -(torch.mean(log_ps) - torch.mean(kl_div))
            self.losslogger = loss

            # The loss should generally decrease with number of iterations, though it is not
            # guaranteed to decrease monotonically because at each iteration the set of
            # context points changes randomly.
            if iteration % 200 == 0:
                print("Iteration " + str(iteration) + ":, Loss = {:.3f}".format(loss.item()))
                # We can set testing = True if we want to check that we are not overfitting.
                if testing:
                    _, predict_train_mean, predict_train_var = self.predict(x_tot_context,
                                                                            y_tot_context,
                                                                            x_tot_context)
                    predict_train_mean = np.squeeze(predict_train_mean.data.numpy(), axis=0)
                    predict_train_var = np.squeeze(predict_train_var.data.numpy(), axis=0)

                    x_test = torch.unsqueeze(x_test, dim=0)
                    _, predict_test_mean, predict_test_var = self.predict(x_tot_context,
                                                                          y_tot_context,
                                                                          x_test)
                    x_test = torch.squeeze(x_test, dim=0)
                    predict_test_mean = np.squeeze(predict_test_mean.data.numpy(), axis=0)
                    predict_test_var = np.squeeze(predict_test_var.data.numpy(), axis=0)

                    # We transform the standardised predicted and actual y values back to the original data
                    # space
                    y_train_mean_pred = y_scaler.inverse_transform(predict_train_mean)
                    y_train_var_pred = y_scaler.var_ * predict_train_var
                    y_train_untransformed = y_scaler.inverse_transform(y_train)

                    # We transform the standardised predicted and actual y values back to the original data
                    # space
                    y_test_mean_pred = y_scaler.inverse_transform(predict_test_mean)
                    y_test_var_pred = y_scaler.var_ * predict_test_var
                    y_test_untransformed = y_scaler.inverse_transform(y_test)

                    r2_train = r2_score(y_train_untransformed, y_train_mean_pred)
                    rmse_train = mean_squared_error(y_train_untransformed, y_train_mean_pred)

                    r2_test_list = []
                    rmse_test_list = []
                    nlpd_test_list = []

                    for j in range(10):
                        indices = np.random.permutation(y_test_untransformed.shape[0])[0:20]
                        r2_test = r2_score(y_test_untransformed[indices, 0], y_test_mean_pred[indices, 0])
                        rmse_test = np.sqrt(mean_squared_error(y_test_untransformed[indices, 0],
                                                       y_test_mean_pred[indices, 0]))
                        nlpd_test = nlpd(y_test_mean_pred[indices, 0], y_test_var_pred[indices, 0],
                                         y_test_untransformed[indices, 0])
                        r2_test_list.append(r2_test)
                        rmse_test_list.append(rmse_test)
                        nlpd_test_list.append(nlpd_test)

                    r2_test_list = np.array(r2_test_list)
                    rmse_test_list = np.array(rmse_test_list)
                    nlpd_test_list = np.array(nlpd_test_list)

                    print("\nmean R^2 (test): {:.3f} +- {:.3f}".format(np.mean(r2_test_list),
                                                                np.std(r2_test_list) / np.sqrt(len(r2_test_list))))
                    print("mean RMSE (test): {:.3f} +- {:.3f}".format(np.mean(rmse_test_list),
                                                               np.std(rmse_test_list) / np.sqrt(len(rmse_test_list))))
                    print("mean NLPD (test): {:.3f} +- {:.3f}\n".format(np.mean(nlpd_test_list),
                                                                np.std(nlpd_test_list) / np.sqrt(len(nlpd_test_list))))

                    print("R2 score (train) = {:.3f}".format(r2_train))
                    print("RMSE score (train) = {:.3f}".format(rmse_train))

                    if iteration % 1000==0:
                        if plotting:
                            x_c = x_scaler.inverse_transform(np.array(x_train))
                            y_c = y_train_untransformed
                            x_t = x_scaler.inverse_transform(np.array(x_test))
                            y_t = x_t**3

                            plt.figure(figsize=(7, 7))
                            plt.scatter(x_c, y_c, color='red', s=15, marker='o', label="Context points")
                            plt.plot(x_t, y_t, linewidth=1, color='red', label="Ground truth")
                            plt.plot(x_t, y_test_mean_pred, color='darkcyan', linewidth=1, label='Mean prediction')
                            plt.plot(x_t[:, 0], y_test_mean_pred[:, 0] - 1.96 * np.sqrt(y_test_var_pred[:, 0]), linestyle= '-.',
                                     marker = None, color='darkcyan', linewidth=0.5)
                            plt.plot(x_t[:, 0], y_test_mean_pred[:, 0] + 1.96 * np.sqrt(y_test_var_pred[:, 0]), linestyle= '-.',
                                     marker = None, color='darkcyan', linewidth=0.5, label='Two standard deviations')
                            plt.fill_between(x_t[:, 0], y_test_mean_pred[:, 0] - 1.96 * np.sqrt(y_test_var_pred[:, 0]),
                                             y_test_mean_pred[:, 0] + 1.96 * np.sqrt(y_test_var_pred[:, 0]),
                                             color='cyan', alpha=0.2)
                            plt.title('Predictive distribution')
                            plt.ylabel('f(x)')
                            plt.yticks([-40, -20, 0, 20, 40])
                            plt.ylim(-80, 80)
                            plt.xlim(-4, 4)
                            plt.xlabel('x')
                            plt.xticks([-4, -2, 0, 2, 4])
                            plt.legend()
                            plt.savefig('results/anp_1d_reg' + str(iteration) + '.png')

            loss.backward()
            self.optimiser.step()

    def predict(self, x_context, y_context, x_target):
        """
        :param x_context: A tensor of dimensions [batch_size, N_context, x_size].
                          When training N_context is randomly sampled between 3 and N_train;
                          when testing N_context = N_train
        :param y_context: A tensor of dimensions [batch_size, N_context, y_size]
        :param x_target: A tensor of dimensions [N_target, x_size]
        :return dist: The distributions over the predicted outputs y_target
        :return mu: A tensor of dimensionality [batch_size, N_target, output_size]
                    describing the means
                    of the normal distribution.
        :return var: A tensor of dimensionality [batch_size, N_target, output_size]
                     describing the variances of the normal distribution.
        """

        r = self.det_encoder.forward(x_context, y_context, x_target)
        # The latent encoder outputs a distribution over the latent embedding z.
        dists_z, _, _ = self.lat_encoder.forward(torch.cat((x_context, y_context),
                                                          dim=2).float())
        zs = [dist.sample() for dist in dists_z]  # [batch_size, r_size]
        z = torch.cat(zs)
        z = z.view(-1, self.r_size)

        # The input to the decoder is the concatenation of the target x values, the deterministic embedding r and the latent variable z
        # the output is the predicted target y for each value of x.
        dists_y, _, _ = self.decoder.forward(x_target.float(), r.float(), z.float())

        # The input to the decoder is the concatenation of the target x values, the deterministic embedding r and the latent variable z
        # the output is the predicted target y for each value of x.
        dist, mu, sigma = self.decoder.forward(x_target.float(), r.float(), z.float())

        return dist, mu, sigma


