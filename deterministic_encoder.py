"""
Function for deterministically encoding context points (x, y)_i using a fully connected neural network.
Input = (x, y)_i; output = r_i.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import *

class DeterministicEncoder(nn.Module):
    """The Deterministic Encoder."""
    def __init__(self, x_size, y_size, r_size, encoder_n_hidden,
                 encoder_hidden_size, self_att=False, cross_att=True,
                 attention_type="uniform"):
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
        self.input_size = x_size + y_size
        self.x_size = x_size
        self.y_size = y_size
        self.r_size = r_size
        self.n_hidden = encoder_n_hidden
        self.hidden_size = encoder_hidden_size
        self.self_att = self_att
        self.cross_att = cross_att
        self.attention_type = attention_type

        if self.cross_att == True:
            if attention_type == "multihead":
                print("Using multihead cross attention.")
                self.cross_attention = MultiHeadAttention(key_size=x_size, value_size=r_size, num_heads=4,
                                                          key_hidden_size=self.r_size, normalise=True)
            else:
                print("Using uniform cross attention.")

        else:
            print("Not using cross attention.")

        if self.self_att == True:
            print("Using multihead self attention.")
            self.self_attention = SelfAttention(input_size=self.input_size, hidden_size=self.hidden_size,
                                                n_hidden=self.n_hidden, output_size=self.r_size, num_heads=4)

        else:
            print("Not using self attention.")
            self.fcs = nn.ModuleList()
            for i in range(self.n_hidden + 1):
                if i == 0:
                    self.fcs.append(nn.Linear(self.input_size, self.hidden_size))

                elif i == self.n_hidden:
                    self.fcs.append(nn.Linear(self.hidden_size, self.r_size))

                else:
                    self.fcs.append(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, x, y, x_target):
        """
        :param x: A tensor of dimensions [batch_size, number of context points
                  N_context, x_size]. In this case each value of x is the concatenation
                  of the input x with the output y
        :return: The embeddings, a tensor of dimensionality [batch_size, N_context,
                 r_size]
        """
        input = torch.cat((x, y), dim=-1).float()    #[batch_size, N_context, (x_size + y_size)]
        batch_size = input.shape[0]
        input = input.view(-1, self.input_size)     #[batch_size * N_context, (x_size + y_size)]

        if self.self_att == True:
            input = self.self_attention.forward(input, batch_size)  #[batch_size, N_context, r_size]

        else:
            #Pass (x, y)_i through the MLP to get r_i.
            for fc in self.fcs[:-1]:
                input = F.relu(fc(input))                #[batch_size * N_context, hidden_size]
            input = self.fcs[-1](input)    #[batch_size * N_context, self.r_size]

        # Aggregate the embeddings
        input = input.view(batch_size, -1, self.r_size)  # [batch_size, N_context, self.r_size]

        #Using cross attention
        if self.attention_type == "multihead":
            output = self.cross_attention.forward(queries=x_target.float(), keys=x.float(), values=input) #[batch_size, N_target, r_size]

        elif self.attention_type == "uniform":
            output = uniform_attention(queries=x_target.float(), values=input)

        elif self.attention_type == "laplace":
            output = laplace_attention(queries=x_target.float(), keys=x.float(), values=input, scale=1.0, normalise=True)

        elif self.attention_type == "dot_product":
            output = dot_product_attention(queries=x_target.float(), keys=x.float(), values=input, normalise=True)

        #Otherwise take the mean of the embeddings as for the vanilla NP (same as uniform).
        else:
            output = torch.squeeze(torch.mean(input, dim=1), dim=1)   #[batch_size, self.r_size]
            output = torch.unsqueeze(output, dim=1).repeat(1, x_target.shape[1], 1)   #[batch_size, N_target, self.r_size]

        return output
