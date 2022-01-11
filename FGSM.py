#!/usr/bin/env python3 
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=0.025, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        pred_adv = self.net(x_adv)
        
        if targeted:
            err = self.criterion(pred_adv, y)
        else:
            err = -self.criterion(pred_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        err.backward()
        
        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)


        pred = self.net(x)
        pred_adv = self.net(x_adv)

        return x_adv, pred_adv, pred
