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

    def __init__(self, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        # Weight initialisation
        init_range = lambda fan_in, fan_out : np.sqrt(6 / (fan_in + fan_out))
        init_Wgx = init_Wix = init_Wfx = init_Wox = init_range(num_hidden, input_dim)
        init_Wgh = init_Wih = init_Wfh = init_Woh = init_range(num_hidden, num_hidden)

        self.Wgx = nn.Parameter(torch.Tensor(input_dim, num_hidden).uniform_(-init_Wgx, init_Wgx))
        self.Wgh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).uniform_(-init_Wgh, init_Wgh))

        self.Wix = nn.Parameter(torch.Tensor(input_dim, num_hidden).uniform_(-init_Wix, init_Wix))
        self.Wih = nn.Parameter(torch.Tensor(num_hidden, num_hidden).uniform_(-init_Wih, init_Wih))

        self.Wfx = nn.Parameter(torch.Tensor(input_dim, num_hidden).uniform_(-init_Wfx, init_Wfx))
        self.Wfh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).uniform_(-init_Wfh, init_Wfh))

        self.Wox = nn.Parameter(torch.Tensor(input_dim, num_hidden).uniform_(-init_Wox, init_Wox))
        self.Woh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).uniform_(-init_Woh, init_Woh))

        self.Wph = nn.Parameter(torch.Tensor(num_hidden, num_classes).zero_())

        # initialise biases
        self.bg = nn.Parameter(torch.Tensor(num_hidden).zero_())
        self.bi = nn.Parameter(torch.Tensor(num_hidden).zero_())
        self.bf = nn.Parameter(torch.Tensor(num_hidden).zero_())
        self.bo = nn.Parameter(torch.Tensor(num_hidden).zero_())

        self.bp = nn.Parameter(torch.Tensor(num_classes).zero_())

        # Weights and biases to device
        self.to(device)

        # initialise cell state and hidden layer
        self.state = torch.Tensor(batch_size, num_hidden).zero_().to(device)
        self.hidden = torch.Tensor(batch_size, num_hidden).zero_().to(device)


    def forward(s, x):
        # shape input x:  (sequence length, batch size, input size=1)
        # shape output p: (batch size, class)
        


        for t in range(x.shape[0]):
            # Gates
            g = torch.tanh(x[t] @ s.Wgx + s.hidden @ s.Wgh + s.bg)
            i = torch.sigmoid(x[t] @ s.Wix + s.hidden @ s.Wih + s.bi)
            f = torch.sigmoid(x[t] @ s.Wfx + s.hidden @ s.Wfh + s.bf)
            o = torch.sigmoid(x[t] @ s.Wox + s.hidden @ s.Woh + s.bo)
            # update state and hidden layer
            s.state = g * i + s.state * f
            s.hidden = torch.tanh(s.state) * o

        s.state.detach_()
        s.hidden.detach_()

        # Compute output only in last layer
        out = s.hidden @ s.Wph + s.bp

        return out
