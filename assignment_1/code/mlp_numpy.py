"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    """
    
    # Define model
    n_in = [n_inputs] + n_hidden
    n_out = n_hidden + [n_classes]
    self.model = [LinearModule(n_in[0], n_out[0])]
    for i in range(1, len(n_hidden)+1):
      self.model += [ReLUModule(), LinearModule(n_in[i], n_out[i])]
    self.model += [SoftMaxModule()]

    self.linearModules = [layer for layer in self.model if isinstance(layer, LinearModule)]

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    
    out = x
    for layer in self.model:
      out = layer.forward(out)
    
    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    """
    
    for layer in self.model[::-1]:
      dout = layer.backward(dout)

    return

  def __str__(self):
    return (str([str(layer) for layer in self.model]))
