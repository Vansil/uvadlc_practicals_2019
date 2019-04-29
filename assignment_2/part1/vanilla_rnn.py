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

    def __init__(self, embed_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()

        # Store parameters
        self.num_hidden = num_hidden
        self.device = device

        # Embedding layer
        self.embedding = nn.Embedding(10, embed_dim)

        # Orthogonal weight initialisation
        self.W_hx = nn.Parameter(torch.Tensor(embed_dim, num_hidden))
        self.W_hh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.W_ph = nn.Parameter(torch.Tensor(num_hidden, num_classes))

        nn.init.orthogonal_(self.W_hx)
        nn.init.orthogonal_(self.W_hh)
        nn.init.orthogonal_(self.W_ph)

        # Bias initialisation
        self.b_h = nn.Parameter(torch.Tensor(num_hidden).zero_())
        self.b_p = nn.Parameter(torch.Tensor(num_classes).zero_())

        # set device
        self.to(device)

    def forward(self, x):
        # shape input x:  (sequence length, batch size, input size=1)
        # shape output p: (batch size, class)
        
        # State initialisation
        batch_size = x.shape[1]
        state = torch.Tensor(batch_size, self.num_hidden).zero_().to(self.device)

        # Sequentially input data
        for i in range(x.shape[0]):
            # print(x[i])
            state = torch.tanh(self.embedding.forward(x[i]) @ self.W_hx + state @ self.W_hh + self.b_h)
        
        # Forward output
        state.detach_()
        out = state @ self.W_ph + self.b_p

        return out
