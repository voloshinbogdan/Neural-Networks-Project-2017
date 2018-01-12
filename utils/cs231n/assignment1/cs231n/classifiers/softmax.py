import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_examples = X.shape[0]
  for i in range(0, num_examples):
    scores = X[i].dot(W)
    dscoresdW = X[i][:, np.newaxis].repeat(W.shape[1], axis=1)
    sum = 0
    dsumdscores = np.zeros_like(scores)
    dlossdscores = np.zeros_like(scores)
    for j in range(0, num_classes):
      sum += np.exp(scores[j])
      dsumdscores[j] = np.exp(scores[j])
    loss += -scores[y[i]] + np.log(sum)
    dsumlogdsum = 1.0 / sum
    dsumlogdscores = dsumlogdsum * dsumdscores
    dlossdscores = (1) * dsumlogdscores
    dlossdscores[y[i]] += -1
    dW += dlossdscores * dscoresdW
    

  loss /= num_examples
  dW /= num_examples

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_examples = X.shape[0]
  
  # forward
  scores = X.dot(W)
  sum = np.sum(np.exp(scores), axis=1)

  loss = np.sum(-scores[range(0, num_examples), y] + np.log(sum))
  loss /= num_examples
  loss += reg * np.sum(W * W)

  # backward
  dsumdscores = np.exp(scores)

  dlossdscores = (1.0 / sum)[:, np.newaxis] * dsumdscores
  dlossdscores[range(0, num_examples), y] -= 1
  dlossdscores /= num_examples

  dW = X.T.dot(dlossdscores)
  dW += reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

