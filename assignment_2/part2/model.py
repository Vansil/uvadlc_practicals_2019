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

import torch.nn as nn
import torch
import numpy as np


class TextGenerationModel(nn.Module):

    def __init__(self, dim_embed, max_seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):
        super(TextGenerationModel, self).__init__()
        
        # Character embedding
        self.embedding = nn.Embedding(vocabulary_size, dim_embed).to(device)

        # LSTM
        self.hidden = None
        self.cell = None
        self.lstm = nn.LSTM(dim_embed, lstm_num_hidden, num_layers=lstm_num_layers)
        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = lstm_num_hidden

        # Final linear layer
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

        # Move model to device
        self.device = device
        self.to(device)

    def forward(self, x):
        # Shape x: [sequence, batch, feature]
        batch_size = x.shape[1]

        # Embed each character
        emb = self.embedding(x)

        # Forward LSTM, update hidden and cell state
        state_shape = [self.lstm_num_layers, batch_size, self.lstm_num_hidden]
        self.reset_state(state_shape)
        lstm_outs, (self.hidden, self.cell) = self.lstm(emb, (self.hidden, self.cell))

        # Forward final linear layer
        outs = self.linear(lstm_outs)

        # Return output at each time step
        return outs

    def reset_state(self, state_shape):
        # Sets hidden and cell state to zeros
        self.hidden = torch.zeros(*state_shape).to(self.device)
        self.cell = torch.zeros(*state_shape).to(self.device)

    def predict(self, cs, length=30, temperature=0):
        # Predict sentence from given character ids
        with torch.no_grad():
            out = cs

            state_shape = [self.lstm_num_layers, 1, self.lstm_num_hidden]
            self.reset_state(state_shape)

            # Add input characters
            for c in cs:
                x = torch.LongTensor([c]).unsqueeze(0).to(self.device)
                o = self.forward(x)

            # Iteratively append next character
            for i in range(length):
                # sample character from last output
                if temperature == 0: # greedy
                    c = o.squeeze().argmax().tolist()
                else:
                    logits = np.exp(o.squeeze().cpu().numpy() / temperature)
                    probs = logits / np.sum(logits)
                    c = np.random.choice(np.arange(len(probs)), p=probs)
                out.append(c)
                # input new character
                x = torch.LongTensor([c]).unsqueeze(0).to(self.device)
                o = self.forward(x)
        
        return out
