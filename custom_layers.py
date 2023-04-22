import numpy as np
import torch.nn as nn
import torch


# RELU
class ReLU(nn.Module):
    def __init__(self, n_slope=0):
        super(ReLU, self).__init__()
        self.n_slope = n_slope

    def forward(self, x):
        """
        Computes the forward pass of the ReLU
        Input:
            -x : Inputs of any shape
        Returns a tuple of: (out,cache)

        The shape on the output is the same as the input

        """
        if self.training:
            output = torch.max(self.n_slope * x, x)
            self.mask = (x > 0).float()

        else:
            output = x

        return output

    def backward(self, output_gradient):
        """
        Computes the backward pass of ReLU

        Input:
            - dout: grads of any shape
            - cache : previous input (used on o forward pass)
        """
        # # init dx and x
        # dx, x = None, cache

        # # zeros all the dx for negative x
        # dx = dout * (x > 0)

        # return dx  # terun gradient

        if self.training:
            input_gradient = output_gradient * self.mask

        else:
            input_gradient = output_gradient
        return input_gradient


# Dropout
class Dropout(nn.Module):
    def __init__(self, dropout_rate):
        super(Dropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, input):
        # self.mask = torch.tensor(
        #     torch.bernoulli(torch.ones_like(input) * (1 - self.dropout_rate))
        # ).to(input.device)
        # # Multiply the input by the mask to "drop out" some values
        # self.output = input * self.mask
        # return self.output

        if self.training:
            # binary mask of shape x
            mask = torch.rand_like(input) > self.dropout_rate

            output = input * mask / (1 - self.dropout_rate)
            # for backward
            self.mask = mask
        else:
            output = input
        return output

    def backward(self, output_gradient, learning_rate):
        # return output_gradient * self.mask

        if self.training:
            input_gradient = output_gradient * self.mask / (1 - self.dropout_rate)
        else:
            input_gradient = output_gradient
        return input_gradient


# Dense
class Dense(nn.Module):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


# reshape
class Reshape(nn.Module):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    

class Softmax(nn.Module):

    def __init__(self):
        super(Softmax,self).__init__()
        self.input = input
    def forward(self,input):
        numerator = torch.exp(input)
        self.output = numerator/torch.sum(numerator)
        return self.output
    
    def backward(self, output_gradient,learning_rate):
        n=self.output.size()[0]
        numerator = torch.tile(self.output,(n,))
        return torch.dot(numerator*(torch.eye(n) - torch.transpose(numerator, 0, 1)), output_gradient)    