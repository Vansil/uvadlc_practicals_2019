"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample
    """
    
    self.params = {
      'weight': np.random.normal(0, .0001, (out_features, in_features)), 
      'bias': np.zeros((out_features, 1))}
    self.grads = {
      'weight': np.zeros((out_features, in_features)), 
      'bias': np.zeros((out_features, 1))}

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module (batchsize, inputsize)
    Returns:
      out: output of the module (batchsize, outputsize)                                                         #
    """
    
    out = x @ self.params['weight'].T + self.params['bias'].T
    self.last_x = x

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    """

    # Compute gradients
    self.grads['weight'] = dout.T @ self.last_x
    self.grads['bias'] = np.sum(dout.T, axis=1).reshape((-1,1))

    dx = dout @ self.params['weight']
    
    return dx

  def __str__(self):
    return "Linear ({} -> {})".format(self.params['weight'].shape[1], self.params['weight'].shape[0])

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    """

    self.negative_x = x < 0
    out = x
    out[self.negative_x] = 0

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    """

    dx = dout
    dx[self.negative_x] = 0
    
    return dx

  def __str__(self):
    return "ReLU"

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module (batchsize, inputsize)
    Returns:
      out: output of the module (batchsize, outputsize)                                                          #
    """

    exp_norm = np.exp(x - x.max(axis=1).reshape((-1,1)))
    out = exp_norm / exp_norm.sum(axis=1).reshape((-1,1))

    self.last_out = out

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    """

    # Compute gradient separately for each individual
    individual_grad = lambda t: np.diag(t) - t.reshape((-1,1)) @ t.reshape((1,-1))
    dout_din = np.apply_along_axis(individual_grad, 1, self.last_out)

    # batch size N, input dim i, output dim o
    dx = np.einsum('Nio,Ni->No', dout_din, dout)

    return dx


  def __str__(self):
    return "SoftMax"

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module (input size)
      y: labels of the input, one-hot (input size)
    Returns:
      out: cross entropy loss (1)
    """

    batch_size = x.shape[0]
    out = np.einsum('ni,ni',y, -np.log(x)) / batch_size

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module (input size)
      y: labels of the input, one-hot (input size)
    Returns:
      dx: gradient of the loss with the respect to the input x. (input size)
    """

    batch_size = x.shape[0]
    dx = y * (-1/x) / batch_size

    return dx

  def __str__(self):
      return "CrossEntropy"
