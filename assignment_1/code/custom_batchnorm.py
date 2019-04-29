import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    self.eps = eps

    # For random gamma init, use: #nn.Parameter(Normal(0,.0001).sample((n_neurons,)))
    self.gamma = nn.Parameter(torch.ones(n_neurons))
    self.beta = nn.Parameter(torch.zeros(n_neurons))


  def forward(self, input):
    """
    Compute the batch normalization, batch statistics are also used at evaluation time
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    """

    assert (input.shape[1] == self.gamma.shape[0]), "Unexpected input dimensions"

    sample_mean = input.mean(dim=0)
    sample_var = input.var(unbiased=False, dim=0)

    out = self.gamma * (input - sample_mean) / torch.sqrt(sample_var + self.eps) + self.beta

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    input = input.double()
    gamma = gamma.double()
    beta = beta.double()

    input_mean_subtr = (input - input.mean(dim=0)) # mean subtracted input
    sample_std = torch.sqrt(input.var(unbiased=False, dim=0) + eps) # sample std including epsilon
    input_norm = input_mean_subtr / sample_std # normalised input
    
    out = gamma * input_norm + beta

    ctx.save_for_backward(input_mean_subtr, sample_std, input_norm, gamma, beta)
    ctx.eps = eps
    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    grad_output = grad_output.double()

    input_mean_subtr, sample_std, input_norm, gamma, beta = ctx.saved_tensors
    eps = ctx.eps

    grad_beta = grad_output.sum(dim=0)
    grad_gamma = torch.einsum('bd,bd->d', grad_output, input_norm)


    B,D = input_mean_subtr.shape
    # x_hat = input_mean_subtr / sample_std
    # grad_xhat = torch.mul(grad_output, gamma)
    # grad_input = 1/B * 1/sample_std * \
    #   (B*grad_xhat - torch.sum(grad_xhat, dim=0) - x_hat * torch.sum(grad_xhat * x_hat, dim=0))




    # compute input gradient per element
    # onehot = lambda l, i: torch.zeros(l).scatter_(0,torch.LongTensor([i]),torch.Tensor([1])).double() # one-hot vector of length l and 1 at index i
    # B,D = input_mean_subtr.shape # number of datapoints in batch (B) and dimensionality of datapoint (D)
    # grad_input = torch.DoubleTensor(B,D)
    # for r in range(B):
    #   for j in range(D):
    #     grad_input[r,j] = torch.einsum('b,b',
    #       grad_output[:,j] * gamma[j],
    #       (onehot(B,r)-1/B)/sample_std[j] - input_norm[:,j]/B/torch.pow(sample_std[j],3) * 
    #         torch.einsum('b,b', input_norm[:,j], onehot(B,r)-1/B))

          # (onehot(B,r)-1/B)/sample_std[j] - input_norm[:,j]/B/torch.pow(sample_std[j],3) * torch.einsum('b,b', input_norm[:,j], onehot(B,r)-1/B)

    raise NotImplementedError

    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones(n_neurons).double())
    self.beta = nn.Parameter(torch.zeros(n_neurons).double())
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    assert (input.shape[1] == self.gamma.shape[0]), "Unexpected input dimensions"
    
    fn = CustomBatchNormManualFunction()
    out = fn.apply(input.double(), self.gamma, self.beta, self.eps)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out
