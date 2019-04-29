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

import numpy as np
import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, embed_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()

        # Store params
        self.num_hidden = num_hidden
        self.device = device

        # Embedding layer
        self.embedding = nn.Embedding(10, embed_dim)

        # Orthogonal weight initialisation
        self.Wgx = nn.Parameter(torch.Tensor(embed_dim, num_hidden))
        self.Wgh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))

        self.Wix = nn.Parameter(torch.Tensor(embed_dim, num_hidden))
        self.Wih = nn.Parameter(torch.Tensor(num_hidden, num_hidden))

        self.Wfx = nn.Parameter(torch.Tensor(embed_dim, num_hidden))
        self.Wfh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))

        self.Wox = nn.Parameter(torch.Tensor(embed_dim, num_hidden))
        self.Woh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))

        self.Wph = nn.Parameter(torch.Tensor(num_hidden, num_classes))

        nn.init.orthogonal_(self.Wgx)
        nn.init.orthogonal_(self.Wgh)
        nn.init.orthogonal_(self.Wix)
        nn.init.orthogonal_(self.Wih)
        nn.init.orthogonal_(self.Wfx)
        nn.init.orthogonal_(self.Wfh)
        nn.init.orthogonal_(self.Wox)
        nn.init.orthogonal_(self.Woh)
        nn.init.orthogonal_(self.Wph)

        # initialise biases
        self.bg = nn.Parameter(torch.Tensor(num_hidden).zero_())
        self.bi = nn.Parameter(torch.Tensor(num_hidden).zero_())
        self.bf = nn.Parameter(torch.Tensor(num_hidden).zero_())
        self.bo = nn.Parameter(torch.Tensor(num_hidden).zero_())

        self.bp = nn.Parameter(torch.Tensor(num_classes).zero_())

        # Weights and biases to device
        self.to(device)


    def forward(s, x):
        # shape input x:  (sequence length, batch size, input size=1)
        # shape output p: (batch size, class)
        
        # initialise cell state and hidden layer
        batch_size = x.shape[1]
        state = torch.Tensor(batch_size, s.num_hidden).zero_().to(s.device)
        hidden = torch.Tensor(batch_size, s.num_hidden).zero_().to(s.device)

        for t in range(x.shape[0]):
            # Embed
            embed = s.embedding(x[t])

            # Gates
            g = torch.tanh(embed @ s.Wgx + hidden @ s.Wgh + s.bg)
            i = torch.sigmoid(embed @ s.Wix + hidden @ s.Wih + s.bi)
            f = torch.sigmoid(embed @ s.Wfx + hidden @ s.Wfh + s.bf)
            o = torch.sigmoid(embed @ s.Wox + hidden @ s.Woh + s.bo)
            # update state and hidden layer
            state = g * i + state * f
            hidden = torch.tanh(state) * o

        # Compute output only in last layer
        state.detach_()
        hidden.detach_()
        out = hidden @ s.Wph + s.bp

        return out
