################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    if config.device == 'best':
        config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_dim, config.num_hidden, \
            config.num_classes, config.batch_size, device)
    else:
        model = LSTM(config.input_dim, config.num_hidden, \
            config.num_classes, config.batch_size, device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = F.cross_entropy
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # Track metrics 
    losses = []
    losses_last10 = []
    accuracies = []
    accuracies_last10 = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Transform input to RNN input format (sequence, batch, input)
        batch_inputs = batch_inputs.t().unsqueeze(2).to(device)
        batch_targets = batch_targets.to(device)

        # Only for time measurement of step through network
        t1 = time.time()

        # forward pass
        logits = model.forward(batch_inputs)

        # backprop
        loss = criterion(logits, batch_targets)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()

        # Compute metrics
        accuracy = (logits.cpu().argmax(dim=1) == batch_targets.cpu()).numpy().mean()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        accuracies_last10.append(accuracy.tolist())
        losses_last10.append(loss.tolist())

        if step % 10 == 0:

            message = "[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss)
            print(message)
            if config.log_path != "":
                with open(config.log_path, "a") as f:
                    f.write(message)
            accuracies.append(np.mean(accuracies_last10))
            losses.append(np.mean(losses_last10))
            accuracies_last10 = []
            losses_last10 = []

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            return losses, accuracies

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="best", help="Training device 'cpu' or 'cuda:0' or 'best'")
    parser.add_argument('--experiment', type=str, default="False", help="Set to true to conduct experiment")
    parser.add_argument('--log_path', type=str, default="", help="Path to log file")

    config = parser.parse_args()

    # Train the model
    if config.experiment == "False":
        train(config)
    else:
        # Conduct experiment
        # Vary input length and evaluate accuracy and loss over time
        file_out = 'output/experiment_results.txt'

        with open(file_out, "a") as f:
            f.write("len;losses;accuracies")

        # run until external time limit
        input_length = 3
        while True:
            print("\n\n\nEXPERIMENT: input length {}\n".format(input_length))
            if config.log_path != "":
                with open(config.log_path, "a") as f:
                    f.write("\n\n\nEXPERIMENT: input length {}\n".format(input_length))
            config.input_length = input_length
            losses, accuracies = train(config)
            # Write to output
            with open(file_out, "a") as f:
                f.write(str(input_length) + ";" + ",".join([str(l) for l in losses]) + ";" + ",".join([str(a) for a in accuracies])+"\n")
            # Set next input size
            input_length += 2
