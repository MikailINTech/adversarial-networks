"""
Created on Wed Jan 23 10:15:27 2019
@author: aamir-mustafa
Implementation Part 2 of Paper: 
    "Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks"  
Here it is not necessary to save the best performing model (in terms of accuracy). The model with high robustness 
against adversarial attacks is chosen.
This coe implements Adversarial Training using PGD Attack.   
"""

#Essential Imports
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils import AverageMeter, Logger, Proximity, Con_Proximity
from model import *



parser = argparse.ArgumentParser("Prototype Conformity Loss Implementation")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 50],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_model', type=float, default=0.005, help="learning rate for model")
parser.add_argument('--lr_prox', type=float, default=0.0001, help="learning rate for Proximity Loss") # as per paper
parser.add_argument('--weight-prox', type=float, default=1, help="weight for Proximity Loss") # as per paper
parser.add_argument('--lr_conprox', type=float, default=0.0001, help="learning rate for Con-Proximity Loss") # as per paper
parser.add_argument('--weight-conprox', type=float, default=0.0001, help="weight for Con-Proximity Loss") # as per paper
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--int-epoch', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay")
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=30)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save-dir', type=str, default='log')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t

def attack(model, criterion, img, label, eps, attack_type, iters):
    adv = img.detach()
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations
        
        noise = 0
        
    for j in range(iterations):
        _,_,out_adv = model(adv.clone())
        loss = criterion(out_adv, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean= torch.mean(torch.abs(adv.grad), dim=1,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=2,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=3,  keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad

        # Optimization step
        adv.data = un_normalize(adv.data) + step * noise.sign()
#        adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + 'CIFAR-10_PC_Loss_PGD_AdvTrain' + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # Data Load
    num_classes=10
    print('==> Preparing dataset')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                             download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, pin_memory=True,
                                              shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, pin_memory=True,
                                             shuffle=False, num_workers=args.workers)
    
# Loading the Model    
    model = Net(adv_output=True).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    criterion_prox_1024 = Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_prox_576 = Proximity(num_classes=num_classes, feat_dim=576, use_gpu=use_gpu)
    
    criterion_conprox_1024 = Con_Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_conprox_576 = Con_Proximity(num_classes=num_classes, feat_dim=576, use_gpu=use_gpu)
    
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    
    optimizer_prox_1024 = torch.optim.SGD(criterion_prox_1024.parameters(), lr=args.lr_prox, weight_decay=5e-04, momentum=0.9)
    optimizer_prox_576 = torch.optim.SGD(criterion_prox_576.parameters(), lr=args.lr_prox, weight_decay=5e-04, momentum=0.9)

    optimizer_conprox_1024 = torch.optim.SGD(criterion_conprox_1024.parameters(), lr=args.lr_conprox, weight_decay=5e-04, momentum=0.9)
    optimizer_conprox_576 = torch.optim.SGD(criterion_conprox_576.parameters(), lr=args.lr_conprox, weight_decay=5e-04, momentum=0.9)
    

    start_time = time.time()

    for epoch in range(args.max_epoch):
        
        adjust_learning_rate(optimizer_model, epoch)
        adjust_learning_rate_prox(optimizer_prox_1024, epoch)
        adjust_learning_rate_prox(optimizer_prox_576, epoch)
        
        adjust_learning_rate_conprox(optimizer_conprox_1024, epoch)
        adjust_learning_rate_conprox(optimizer_conprox_576, epoch)
        
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        if epoch < args.int_epoch:
            train_noCL(model, cross_entropy_loss, 
                          optimizer_model,
                          trainloader, device, num_classes, epoch)
        else:
            train_CL(model, cross_entropy_loss, criterion_prox_1024, criterion_prox_576, 
                  criterion_conprox_1024, criterion_conprox_576, 
                  optimizer_model, optimizer_prox_1024, optimizer_prox_576,
                  optimizer_conprox_1024, optimizer_conprox_576,
                  trainloader, device, num_classes, epoch)

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")   #Tests after every 10 epochs
            acc, err = test(model, testloader, device, num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

            state_ = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer_model': optimizer_model.state_dict(), 'optimizer_prox_1024': optimizer_prox_1024.state_dict(),
                     'optimizer_prox_576': optimizer_prox_576.state_dict(), 'optimizer_conprox_1024': optimizer_conprox_1024.state_dict(),
                     'optimizer_conprox_576': optimizer_conprox_576.state_dict(),}
                     
            torch.save(state_, 'Model_AdvTrain_PGD_customloss.pth')
            
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train_CL(model, cross_entropy_loss, criterion_prox_1024, criterion_prox_576, 
              criterion_conprox_1024, criterion_conprox_576, 
              optimizer_model, optimizer_prox_1024, optimizer_prox_576,
              optimizer_conprox_1024, optimizer_conprox_576,
              trainloader, device, num_classes, epoch):
    
#    model.train()
    xent_losses = AverageMeter() #Computes and stores the average and current value
    prox_losses_1024 = AverageMeter()
    prox_losses_576= AverageMeter()
    
    conprox_losses_1024 = AverageMeter()
    conprox_losses_576= AverageMeter()
    losses = AverageMeter()
    
    #Batchwise training
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)
        model.eval()
        eps= np.random.uniform(0.02,0.05)
        adv = attack(model, cross_entropy_loss, data, labels, eps=eps, attack_type='pgd', iters= 10) # Generates Batch-wise Adv Images
        adv.requires_grad= False
        
        adv= normalize(adv)
        adv= adv.to(device)
        true_labels_adv= labels
        data= torch.cat((data, adv),0)
        labels= torch.cat((labels, true_labels_adv))
        model.train()
        
        feats576, feats1024, outputs = model(data) 
        loss_xent = cross_entropy_loss(outputs, labels)  
        
        loss_prox_1024 = criterion_prox_1024(feats1024, labels) 
        loss_prox_576= criterion_prox_576(feats576, labels)
        
        loss_conprox_1024 = criterion_conprox_1024(feats1024, labels) 
        loss_conprox_576= criterion_conprox_576(feats576, labels)
        
        loss_prox_1024 *= args.weight_prox 
        loss_prox_576 *= args.weight_prox
        
        loss_conprox_1024 *= args.weight_conprox 
        loss_conprox_576 *= args.weight_conprox
        
        loss = loss_xent + loss_prox_1024 + loss_prox_576  - loss_conprox_1024 - loss_conprox_576 # total loss
        optimizer_model.zero_grad()
        
        optimizer_prox_1024.zero_grad()
        optimizer_prox_576.zero_grad()
        
        optimizer_conprox_1024.zero_grad()
        optimizer_conprox_576.zero_grad()

        loss.backward()
        optimizer_model.step() 
        
        for param in criterion_prox_1024.parameters():
            param.grad.data *= (1. / args.weight_prox)
        optimizer_prox_1024.step() 
        
        for param in criterion_prox_576.parameters():
            param.grad.data *= (1. / args.weight_prox)
        optimizer_prox_576.step()
        

        for param in criterion_conprox_1024.parameters():
            param.grad.data *= (1. / args.weight_conprox)
        optimizer_conprox_1024.step() 
        
        for param in criterion_conprox_576.parameters():
            param.grad.data *= (1. / args.weight_conprox)
        optimizer_conprox_576.step()
        
        losses.update(loss.item(), labels.size(0)) 
        xent_losses.update(loss_xent.item(), labels.size(0))
        prox_losses_1024.update(loss_prox_1024.item(), labels.size(0))
        prox_losses_576.update(loss_prox_576.item(), labels.size(0))

        conprox_losses_1024.update(loss_conprox_1024.item(), labels.size(0))
        conprox_losses_576.update(loss_conprox_576.item(), labels.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})  XentLoss {:.6f} ({:.6f})  ProxLoss_1024 {:.6f} ({:.6f}) ProxLoss_576 {:.6f} ({:.6f}) \n ConProxLoss_1024 {:.6f} ({:.6f}) ConProxLoss_576 {:.6f} ({:.6f}) " \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, 
                          prox_losses_1024.val, prox_losses_1024.avg, prox_losses_576.val, prox_losses_576.avg , 
                          conprox_losses_1024.val, conprox_losses_1024.avg, conprox_losses_576.val,
                          conprox_losses_576.avg  ))

def train_noCL(model, cross_entropy_loss, 
              optimizer_model,
              trainloader, device, num_classes, epoch):
    
#    model.train()
    losses = AverageMeter()
    
    #Batchwise training
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)
        model.eval()
        eps= np.random.uniform(0.02,0.05)
        adv = attack(model, cross_entropy_loss, data, labels, eps=eps, attack_type='pgd', iters= 10) # Generates Batch-wise Adv Images
        adv.requires_grad= False
        
        adv= normalize(adv)
        adv= adv.to(device)
        true_labels_adv= labels
        data= torch.cat((data, adv),0)
        labels= torch.cat((labels, true_labels_adv))
        model.train()
        
        _,_, outputs = model(data) 
        loss_xent = cross_entropy_loss(outputs, labels)  
       
        
        loss = loss_xent
        optimizer_model.zero_grad()
        

        loss.backward()
        optimizer_model.step() 
        
        
        losses.update(loss.item(), labels.size(0)) 
   
        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) )" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))

def test(model, testloader, device, num_classes, epoch):
    model.eval()  
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            _,_, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_model'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_model'] = state['lr_model']
            
def adjust_learning_rate_prox(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_prox'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_prox'] = state['lr_prox'] 

def adjust_learning_rate_conprox(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_conprox'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_conprox'] = state['lr_conprox']             
if __name__ == '__main__':
    main()
