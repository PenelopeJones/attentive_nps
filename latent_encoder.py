"""
Function for encoding context points (x, y)_i using latent space.
Input = (x, y)_i; output = r_i.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

class LatentEncoder(nn.Module):
    """The Latent Encoder."""
    def __init__(self, input_size, r_size, encoder_n_hidden, encoder_hidden_size):
        """
        :param input_size: An integer describing the dimensionality of the input to the encoder;
                           in this case the sum of x_size and y_size
        :param r_size: An integer describing the dimensionality of the embedding, r_i
        :param encoder_n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param encoder_hidden_size: An integer describing the number of nodes in each layer of
                                    the neural network
        """

        super().__init__()
        self.input_size = input_size
        self.r_size = r_size
        self.n_hidden = encoder_n_hidden
        self.hidden_size = encoder_hidden_size

        self.n2_hidden = 2
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()

        #Encoder function taking as input (x,y)_i and outputting r_i
        for i in range(self.n_hidden + 1):
            if i == 0:
                self.fcs1.append(nn.Linear(input_size, encoder_hidden_size))

            elif i == encoder_n_hidden:
                self.fcs1.append(nn.Linear(encoder_hidden_size, r_size))

            else:
                self.fcs1.append(nn.Linear(encoder_hidden_size, encoder_hidden_size))

        #For the latent encoder, we also have a second encoder function which takes the aggregated embedding r as
        #an input and outputs a mean and the log(variance)
        for i in range(self.n2_hidden + 1):
            if i ==self.n2_hidden:
                self.fcs2.append(nn.Linear(r_size, 2*r_size))

            else:
                self.fcs2.append(nn.Linear(r_size, r_size))

    def forward(self, x):
        """
        :param x: A tensor of dimensions [batch_size, number of context points
                  N_context, x_size + y_size]. In this case each value of x is the concatenation
                  of the input x with the output y
        :return: The embeddings, a tensor of dimensionality [batch_size, N_context,
                 r_size]
        """
        batch_size = x.shape[0]
        #Pass (x, y)_i through the first MLP.
        x = x.view(-1, self.input_size)
        for fc in self.fcs1[:-1]:
            x = F.relu(fc(x))
        x = self.fcs1[-1](x)

        #Aggregate the embeddings
        x = x.view(batch_size, -1, self.r_size)  #[batch_size, N_context, r_size]
        x = torch.squeeze(torch.mean(x, dim=1), dim=1) #[batch_size, r_size]

        #Pass the aggregated embedding through the second MLP to obtain means and variances parametrising the distribution
        #over the latent variable z.
        for fc in self.fcs2[:-1]:
            x = F.relu(fc(x))
        x = self.fcs2[-1](x)        #[batch_size, 2*r_size]

        # The outputs are the latent variable mean and log(variance)
        mus_z, ws_z = x[:, :self.r_size], x[:, self.r_size:]
        sigmas_z = 0.1 + 0.9 * F.sigmoid(ws_z)   #[batch_size, r_size]

        dists_z = [MultivariateNormal(mu, torch.diag_embed(sigma)) for mu, sigma in
                 zip(mus_z, sigmas_z)]

        return dists_z, mus_z, sigmas_z


