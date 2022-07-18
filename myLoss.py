import torch
import torch.nn as nn

class MaxLikelyloss(nn.Module):
    def __init__(self):
        super(MaxLikelyloss, self).__init__()

    def forward(self, input, target):
        img = input[:,:-1]
        sigma = input[:,-1:] 
        loss = ( (target - input)**2 / torch.exp(sigma) + sigma )/2
        return loss.sum()