"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.nn import *

class Flatten(Module):
  '''Flattens its input, used to switch from conv to linear'''
  def forward(self, x):
    return x.view(x.size()[0], -1)

class ConvNet(Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODONE:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet,self).__init__()

    self.model = Sequential(
      Conv2d(n_channels,64,3,padding=1),# conv1
      BatchNorm2d(64),
      ReLU(),
      MaxPool2d(3,stride=2,padding=1),  # maxpool1
      Conv2d(64,128,3,padding=1),       # conv2
      BatchNorm2d(128),
      ReLU(),
      MaxPool2d(3,stride=2,padding=1),  # maxpool2
      Conv2d(128,256,3,padding=1),      # conv3a
      BatchNorm2d(256),
      ReLU(),
      Conv2d(256,256,3,padding=1),      # conv3b
      BatchNorm2d(256),
      ReLU(),
      MaxPool2d(3,stride=2,padding=1),  # maxpool3
      Conv2d(256,512,3,padding=1),      # conv4a
      BatchNorm2d(512),
      ReLU(),
      Conv2d(512,512,3,padding=1),      # conv4b
      BatchNorm2d(512),
      ReLU(),
      MaxPool2d(3,stride=2,padding=1),  # maxpool4
      Conv2d(512,512,3,padding=1),      # conv5a
      BatchNorm2d(512),
      ReLU(),
      Conv2d(512,512,3,padding=1),      # conv5b
      BatchNorm2d(512),
      ReLU(),
      MaxPool2d(3,stride=2,padding=1),  # maxpool5
      Flatten(),
      Linear(512,n_classes)  # linear
    )
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODONE:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.model.forward(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
