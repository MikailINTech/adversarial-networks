#!/usr/bin/env python3 
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def pgd(self, x, y, iterations = 10, targeted=False, eps=0.025, x_val_min=0, x_val_max=1, infer = True):
        images = x.to(device)
        labels = y.to(device)
        
        ori_images = images.data
        
        for _ in range(iterations):
            images.requires_grad = True
            _,_,pred_adv = self.net(images)
            
            if targeted:
                err = self.criterion(pred_adv, labels)
            else:
                err = -self.criterion(pred_adv, labels)
            
            self.net.zero_grad()
            err.backward()
            
            x_adv = images - eps*images.grad.sign()
            eta = torch.clamp(x_adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, x_val_min, x_val_max).detach_()
            
            
        if infer:
            _,_,pred = self.net(x)
            _,_,pred_adv = self.net(x_adv)
    
            return x_adv, pred_adv, pred
        else:
            return x_adv
    
    def fgsm(self, x, y, targeted=False, eps=0.025, x_val_min=0, x_val_max=1, infer = True):
        return self.pgd(x, y, iterations = 1, targeted=targeted, eps=eps, x_val_min=x_val_min, x_val_max=x_val_max,infer=infer)
