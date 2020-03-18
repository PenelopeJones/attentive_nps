import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    """

    """
    def __init__(self, key_size, value_size, num_heads, key_hidden_size, normalise=True):
        """

        :param num_heads:
        :param normalise:
        """
        super().__init__()
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._key_hidden_size = key_hidden_size
        self._head_size = int(self._value_size / self._num_heads)
        self._normalise = normalise

        self._query_transform = nn.Linear(self._key_size, self._num_heads*self._key_hidden_size, bias=False)   #Apply linear transformation [key_size, key_hidden_size]
        self._key_transform = nn.Linear(self._key_size, self._num_heads*self._key_hidden_size, bias=False)  #Apply linear transformation [key_size, key_hidden_size]
        self._value_transform = nn.Linear(self._value_size, self._num_heads*self._head_size, bias=False)  #Apply linear transformation [value_size, head_size]
        self._head_transform = nn.Linear(self._num_heads*self._head_size, self._value_size, bias=False)   #Apply final linear transformation [num_heads*head_size, value_size]

    def forward(self, queries, keys, values):
        """

        :param queries: [batch_size, N_target, key_size]
        :param keys: [batch_size, N_context, key_size]
        :param values:
        :return:
        """
        self._batch_size = queries.shape[0]
        self._n_target = queries.shape[1]
        self._n_context = keys.shape[1]

        #Linearly transform the queries, keys and values:
        queries = self._query_transform(queries).view(self._batch_size, self._n_target, self._num_heads, self._key_hidden_size)
        keys = self._key_transform(keys).view(self._batch_size, self._n_context, self._num_heads, self._key_hidden_size)
        values = self._value_transform(values).view(self._batch_size, self._n_context, self._num_heads, self._head_size)

        #Transpose so that in form [batch_size, num_heads, ...]
        queries = queries.transpose(1, 2)   #[batch_size, num_heads, N_target, key_hidden_size]
        keys = keys.transpose(1, 2)         #[batch_size, num_heads, N_context, key_hidden_size]
        values = values.transpose(1, 2)     #[batch_size, num_heads, N_context, head_size]

        attention = dot_product_attention(queries, keys, values, normalise=self._normalise)  #[batch_size, num_heads, N_target, head_size]

        attention = attention.transpose(1, 2) #[batch_size, N_target, num_heads, head_size]
        attention = attention.reshape(self._batch_size, self._n_target, -1) #[batch_size, N_target, num_heads*head_size]
        output = self._head_transform(attention) #[batch_size, N_target, value_size]

        return output


def uniform_attention(queries, values):
    """
    In the case of uniform attention, the weight assigned to each value is independent of the value of the
    corresponding key; we can simply take the average of all of the values. This is the equivalent of the "vanilla"
    neural process, where r* is the average of the context set embeddings.

    :param queries: Queries correspond to x_target. [batch_size, N_target, key_size]
    :param values: Values corresponding to the aggregated embeddings r_i. [batch_size, N_context, value_size]
    :return:
    """

    N_target = queries.shape[1]
    attention= torch.mean(values, dim=1, keepdim=True) #[batch_size, 1, value_size]
    output= attention.repeat(1, N_target, 1)  #[batch_size, N_target, value_size]

    return output

def laplace_attention(queries, keys, values, scale, normalise=True):
    """
    Here we compute the Laplace exponential attention. Each value is weighted by an amount that depends
    on the distance of the query from the corresponding key (specifically, w_i ~ exp(-||q-k_i||/scale))
    :param queries: e.g. query corresponding to x_target: [batch_size, N_target, key_size]
    :param keys: e.g. x_context: [batch_size, N_context, key_size]
    :param values: e.g. values corresponding to the aggregated embeddings r_i [batch_size, N_context, value_size]
    :param scale: float value which scales the L1 distance.
    :param normalise: Boolean, determines whether we should normalise s.t. sum of weights = 1.
    :return: A tensor [batch_size, N_target, value_size].
    """

    keys = torch.unsqueeze(keys, dim=1) #[batch_size, 1, N_context, key_size]
    queries = torch.unsqueeze(queries, dim=1) #[batch_size, N_target, 1, key_size]

    unnorm_weights = -torch.abs((keys - queries)/scale) #[batch_size, N_target, N_context, key_size]
    unnorm_weights = torch.sum(unnorm_weights, dim=-1, keepdim=False) #[batch_size, N_target, N_context]

    if normalise:
        attention = torch.softmax(unnorm_weights, dim=-1)   #[batch_size, N_target, N_context]
    else:
        attention = 1 + torch.tanh(unnorm_weights)    #[batch_size, N_target, N_context]

    #Einstein summation over weights and values
    output= torch.matmul(attention, values)  #[batch_size, N_target, value_size]

    return output


def dot_product_attention(queries, keys, values, normalise=True):
    """

    :param queries:[batch_size, N_target, key_size]
    :param keys:[batch_size, N_context, key_size]
    :param values: []
    :param normalise:
    :return:
    """
    key_size = keys.shape[-1]
    scale = np.sqrt(key_size)

    unnorm_weights = torch.matmul(queries, keys.transpose(-2, -1)) / scale #[batch_size, N_target, N_context]

    if normalise:
        attention = torch.softmax(unnorm_weights, dim=-1)

    else:
        attention = torch.sigmoid(unnorm_weights)  #[batch_size, N_target, N_context]

    # Einstein summation over weights and values
    output = torch.matmul(attention, values)  # [batch_size, N_target, value_size]
    return output






    

