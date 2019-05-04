# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TextDataset
from model import TextGenerationModel
from summary import Writer

################################################################################

def train(config):
    # Initialise custom summary writer
    writer = Writer(config.summary_path)

    # Initialize the dataset and data loader
    writer.log("Loading dataset from: " + config.txt_file)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the device which to run the model on
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    writer.log("Device: " + device)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.lstm_num_embed, config.seq_length, dataset.vocab_size,
                 config.lstm_num_hidden, config.lstm_num_layers, device)
    if config.checkpoint_path is not None:
        model = torch.load(config.checkpoint_path).to(device)
    writer.log("Model:\n" + str(model))


    # Setup the loss and optimizer
    criterion = F.cross_entropy
    learning_rate = config.learning_rate
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    epoch = 0
    total_step = 0
    while True:
        

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            batch_inputs = torch.LongTensor([x.tolist() for x in batch_inputs]).to(device)
            batch_targets = torch.LongTensor([x.tolist() for x in batch_targets]).to(device)

            # Forward pass
            logits = model.forward(batch_inputs)

            # backprop
            optimizer.zero_grad()
            loss = criterion(logits.transpose(1,2), batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            with torch.no_grad():
                # accuracy = fraction of characters predicted correctly
                accuracy = (logits.argmax(dim=2) == batch_targets).to(dtype=torch.float).mean()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            # Learning rate decay
            if step % config.learning_rate_step == 0 and step != 0:
                learning_rate *= config.learning_rate_decay
                writer.log("Reduced learning rate: {}".format(learning_rate))
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate

            # Metrics and samples
            if step % config.print_every == 0:
                writer.log("[{}] Epoch {:02d}, Train Step {:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        epoch, step, config.batch_size, examples_per_second,
                        accuracy, loss
                ))
                writer.write('metrics', '{},{},{},{}'.format(total_step, accuracy, loss, learning_rate))

            if step % config.sample_every == 0:
                writer.log("Generating sentences")
                writer.write('samples', 'ITER{}'.format(step))
                for temp in [0, .05, .2, .5]:
                    writer.log("\nTemperature: {}".format(temp))
                    writer.write('samples', 'T{}'.format(temp))
                    for i in np.random.choice(dataset.vocab_size, size=5):
                        text = dataset.convert_to_string(model.predict([i], 100, temp)).replace("\n", "<br>")
                        writer.log(text)
                        writer.write('samples', text)
                    for string in ["1:1. In the beginning God created", "1:5. And he called the light Day, and the darkness", 
                        "7:1. And the Lord said to him:", "Genesis Chapter 7"]:
                        text = dataset.convert_to_string(model.predict(dataset.convert_to_id(string), 100, temp)).replace("\n", "<br>")
                        writer.log(text)
                        writer.write('samples', text)

            if step % config.checkpoint_every == 0:    
                writer.save_model(model, step)

            # if step == config.train_steps:
            #     # If you receive a PyTorch data-loader error, check this bug report:
            #     # https://github.com/pytorch/pytorch/pull/9655
            #     break

            total_step += 1
        epoch += 1

    writer.log('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='data/bibleFull.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_embed', type=str, default=32, help='Dimensionality of the model character embeddings')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="summaries/default", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=1000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=5000, help='How often to sample from the model and save it')
    parser.add_argument('--checkpoint_every', type=int, default=5000, help='How often to sample from the model and save it')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint file')

    config = parser.parse_args()
    config.train_steps = int(config.train_steps)

    # Train the model
    train(config)
