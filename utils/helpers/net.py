from builtins import range
from builtins import object
import numpy as np

# TODO: Add mode train/test variable
class NeuralNetwork(object):

    def __init__(self, input_layer, loss_layer, loss_name, ground_truth, mode_name, params, grads):
        self.input_layer = input_layer
        self.loss_layer = loss_layer
        self.loss_name = loss_name
        self.ground_truth = ground_truth
        self.mode_name = mode_name
        self.params = params
        self.grads = grads

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        scores = None

        x = {
            '#input': X,
            self.mode_name: mode,
            self.ground_truth: y,
            self.loss_name: 0
        }
        scores = self.input_layer.forward(x)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        self.params[self.loss_name] = 0
        self.grads.clear()

        self.loss_layer.backward(self.loss_layer.cache)

        return self.params[self.loss_name], self.grads
