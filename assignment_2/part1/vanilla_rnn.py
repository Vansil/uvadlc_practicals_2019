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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        # Weight initialisation
        init_range = lambda fan_in, fan_out : np.sqrt(6 / (fan_in + fan_out))
        init_Whx = init_range(num_hidden, input_dim)
        init_Whh = init_range(num_hidden, num_hidden)

        self.W_hx = nn.Parameter(torch.Tensor(input_dim, num_hidden).uniform_(-init_Whx, init_Whx))
        self.W_hh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).uniform_(-init_Whh, init_Whh))
        self.W_ph = nn.Parameter(torch.Tensor(num_hidden, num_classes).zero_())

        # Bias initialisation
        self.b_h = nn.Parameter(torch.Tensor(num_hidden).zero_())
        self.b_p = nn.Parameter(torch.Tensor(num_classes).zero_())

        # State initialisation
        self.state = torch.Tensor(batch_size, num_hidden).zero_().to(device)

        # set device
        self.to(device)

    def forward(self, x):
        # shape input x:  (sequence length, batch size, input size=1)
        # shape output p: (batch size, class)
        
        for i in range(x.shape[0]):
            self.state = torch.tanh(x[i] @ self.W_hx + self.state @ self.W_hh + self.b_h)
        
        self.state.detach_()
        out = self.state @ self.W_ph + self.b_p

        return out
