"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch.nn.functional as F
import torch.optim as optim
import pickle
import datetime
import matplotlib
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
OUTPUT_DIR_DEFAULT = './output'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  """

  correct_prediction = predictions.argmax(dim=1) - targets.argmax(dim=1) == 0
  accuracy = correct_prediction.sum() / len(correct_prediction)

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []
  
  output_dir    = FLAGS.output_dir
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  
  learning_rate = FLAGS.learning_rate
  max_steps     = FLAGS.max_steps
  batch_size    = FLAGS.batch_size
  eval_freq     = FLAGS.eval_freq
  data_dir      = FLAGS.data_dir

  # Obtain dataset
  dataset = cifar10_utils.get_cifar10(data_dir)
  n_inputs = dataset['train'].images[0].reshape(-1).shape[0]
  n_classes = dataset['train'].labels[0].shape[0]
  n_test = dataset['test'].images.shape[0]

  # Initialise MLP
  net = MLP(n_inputs, dnn_hidden_units, n_classes)
  loss_fn = F.cross_entropy
  print("Network architecture:\n\t{}\nLoss module:\n\t{}".format(str(net), str(loss_fn)))

  # Evaluation vars
  train_loss = []
  gradient_norms = []
  train_acc = []
  test_acc = []
  iteration = 0

  # Training
  optimizer = optim.SGD(net.parameters(), lr=learning_rate)
  while iteration < max_steps:
    iteration += 1

    # Sample a mini-batch
    x, y = dataset['train'].next_batch(batch_size)
    x = torch.from_numpy(x.reshape((batch_size, -1)))
    y = torch.from_numpy(y)

    # Forward propagation
    prediction = net.forward(x)
    loss = loss_fn(prediction, y)
    acc = accuracy(prediction, y)
    train_acc.append( (iteration, acc) )
    train_loss.append( (iteration, loss) )

    # Backprop
    optimizer.zero_grad()
    loss.backward()

    # Weight update in linear modules
    optimizer.step()
    norm = 0
    for params in net.parameters:
      if len(params.shape) == 1:
        equation = 'ij,ij'
      else:
        equation = 'i,i'
      norm += np.einsum(equation, params, params)
    gradient_norms.append( (iteration, norm.reshape(-1)) )

    # Evaluation
    if iteration % eval_freq == 0:
      x = torch.from_numpy(dataset['test'].images.reshape((n_test,-1)))
      y = torch.from_numpy(dataset['test'].labels)
      prediction = net.forward(x)
      acc = accuracy(prediction, y)
      test_acc.append( (iteration, acc) )
      print("Iteration: {}\t\tTest accuracy: {}".format(iteration, acc))
  
  # Save raw output
  now = datetime.datetime.now()
  time_stamp = "{}{}{}{}{}".format(now.year, now.month, now.day, now.hour, now.minute)
  metrics = {"train_loss": train_loss,
             "gradient_norms": gradient_norms,
             "train_acc": train_acc,
             "test_acc": test_acc}
  raw_data = {"net": net,
              "metrics": metrics}
  pickle.dump(raw_data, open(os.path.join(output_dir, "torch_raw_data_" + time_stamp), "wb"))

  # Save plots
  # Loss
  fig, ax = plt.subplots()
  iter = [i for (i,q) in train_loss]
  loss = [q for (i,q) in train_loss]
  ax.plot(iter, loss)
  ax.set(xlabel='Iteration', ylabel='Loss (log)',
        title='Batch training loss')
  ax.set_yscale('log')
  ax.grid()
  fig.savefig(os.path.join(output_dir, "torch_loss_" + time_stamp + ".png"))
  # gradient norm
  fig, ax = plt.subplots()
  iter = [i for (i,q) in gradient_norms]
  norm = [q for (i,q) in gradient_norms]
  ax.plot(iter, norm)
  ax.set(xlabel='Iteration', ylabel='Norm',
        title='Gradient norm')
  ax.grid()
  fig.savefig(os.path.join(output_dir, "torch_gradient_norm_" + time_stamp + ".png"))
  # accuracies
  fig, ax = plt.subplots()
  iter = [i for (i,q) in train_acc]
  accu = [q for (i,q) in train_acc]
  ax.plot(iter, accu, label='Train')
  iter = [i for (i,q) in test_acc]
  accu = [q for (i,q) in test_acc]
  ax.plot(iter, accu, label='Test')
  ax.set(xlabel='Iteration', ylabel='Accuracy',
        title='Train and test accuracy')
  ax.legend()
  ax.grid()
  fig.savefig(os.path.join(output_dir, "torch_accuracy_" + time_stamp + ".png"))

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--output_dir', type = str, default = OUTPUT_DIR_DEFAULT,
                      help='Directory for saving output data')
  FLAGS, unparsed = parser.parse_known_args()

  main()