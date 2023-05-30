# -*- coding: utf-8 -*- 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Correlation(nn.Module):
    def __init__(self, max_displacement=4, *args, **kwargs):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.output_dim = 2 * self.max_displacement + 1
        self.pad_size = self.max_displacement

    def forward(self, x1, x2):
        B, C, H, W = x1.size()

        x2 = F.pad(x2, [self.pad_size] * 4)
        cv = []
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                cost = x1 * x2[:, :, i:(i + H), j:(j + W)]
                cost = torch.mean(cost, 1, keepdim=True)
                cv.append(cost)
        return torch.cat(cv, 1)


def corr(input1, input2):
    max_displacement = 4
    output_dim = 2 * max_displacement + 1
    pad_size = max_displacement

    B, C, H, W = input1.size()

    input2 = F.pad(input2, [pad_size] * 4)
    cv = []
    for i in range(output_dim):
        for j in range(output_dim):
            cost = input1 * input2[:, :, i:(i + H), j:(j + W)]
            cost = torch.mean(cost, 1, keepdim=True)
            cv.append(cost)
    return torch.cat(cv, 1)
    

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]