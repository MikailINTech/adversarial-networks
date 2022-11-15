"""
Original file : 
@author: aamir-mustafa
This file :
@author: MikailINTech

Implementation Part 2 of Paper: 
    "Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks"  

Here it is not necessary to save the best performing model (in terms of accuracy). The model with high robustness 
against adversarial attacks is chosen.  
"""

# Essential Imports
import os
import sys
import argparse
import time
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils import AverageMeter, Logger, Proximity, Con_Proximity
from model import Net, get_train_loader, get_validation_loader, test_natural
from attacks import Attack

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


now = datetime.now()
logdir = "runs/" + now.strftime("%Y%m%d-%H") + "/"

writer = SummaryWriter(logdir)

parser = argparse.ArgumentParser("Prototype Conformity Loss Implementation")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('-bs', '--batch-size', type=int, default=64)
parser.add_argument('-vs', '--valid-size', type=int, default=1024)
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 50],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_model', type=float, default=0.001,
                    help="learning rate for model")
parser.add_argument('--lr_prox', type=float, default=0.0001,
                    help="learning rate for Proximity Loss")  # as per paper
parser.add_argument('--weight-prox', type=float, default=1,
                    help="weight for Proximity Loss")  # as per paper
parser.add_argument('--lr_conprox', type=float, default=0.0001,
                    help="learning rate for Con-Proximity Loss")  # as per paper
parser.add_argument('--weight-conprox', type=float, default=0.0001,
                    help="weight for Con-Proximity Loss")  # as per paper
parser.add_argument('--max-epoch', type=int, default=10)
parser.add_argument('--int-epoch', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.1,
                    help="learning rate decay")
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=500)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=24)
parser.add_argument('--save-dir', type=str, default='log')

args = parser.parse_args()

torch.manual_seed(args.seed)

state = {k: v for k, v in args._get_kwargs()}

valid_size = args.valid_size
batch_size = args.batch_size


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' +
                        'CIFAR-10_PC_Loss_PGD_AdvTrain' + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")


    # Data Load
    num_classes = 10
    print('==> Preparing dataset')
    transform_train = transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), ])

    trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                            download=True, transform=transform_train)

    trainloader = get_train_loader(
        trainset, batch_size=batch_size, valid_size=valid_size)

    testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                           download=True, transform=transform_test)

    testloader = get_validation_loader(testset, batch_size, valid_size)

# Loading the Model
    model = Net(pretrain=True,adv_output=True).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()

    criterion_prox_1024 = Proximity(
        num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_prox_576 = Proximity(
        num_classes=num_classes, feat_dim=576, use_gpu=use_gpu)

    criterion_conprox_1024 = Con_Proximity(
        num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_conprox_576 = Con_Proximity(
        num_classes=num_classes, feat_dim=576, use_gpu=use_gpu)

    optimizer_model = torch.optim.SGD(
        model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

    optimizer_prox_1024 = torch.optim.SGD(criterion_prox_1024.parameters(
    ), lr=args.lr_prox, weight_decay=5e-04, momentum=0.9)
    optimizer_prox_576 = torch.optim.SGD(criterion_prox_576.parameters(
    ), lr=args.lr_prox, weight_decay=5e-04, momentum=0.9)

    optimizer_conprox_1024 = torch.optim.SGD(criterion_conprox_1024.parameters(
    ), lr=args.lr_conprox, weight_decay=5e-04, momentum=0.9)
    optimizer_conprox_576 = torch.optim.SGD(criterion_conprox_576.parameters(
    ), lr=args.lr_conprox, weight_decay=5e-04, momentum=0.9)

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
            print("==> Test")  # Tests after every 10 epochs
            acc = test_natural(model, testloader)
            print("Accuracy (%): {}".format(acc))

            state_ = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                      'optimizer_model': optimizer_model.state_dict(), 'optimizer_prox_1024': optimizer_prox_1024.state_dict(),
                      'optimizer_prox_576': optimizer_prox_576.state_dict(), 'optimizer_conprox_1024': optimizer_conprox_1024.state_dict(),
                      'optimizer_conprox_576': optimizer_conprox_576.state_dict(), }

            torch.save(state_, 'adv_trained_model.pth')

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    writer.close()


def train_CL(model, cross_entropy_loss, criterion_prox_1024, criterion_prox_576,
             criterion_conprox_1024, criterion_conprox_576,
             optimizer_model, optimizer_prox_1024, optimizer_prox_576,
             optimizer_conprox_1024, optimizer_conprox_576,
             trainloader, device, num_classes, epoch):

    #    model.train()
    xent_losses = AverageMeter()  # Computes and stores the average and current value
    prox_losses_1024 = AverageMeter()
    prox_losses_576 = AverageMeter()

    conprox_losses_1024 = AverageMeter()
    conprox_losses_576 = AverageMeter()
    losses = AverageMeter()
    
    att = Attack(model,cross_entropy_loss)

    # Batchwise training
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)
        model.eval()
        adv = att.pgd(data, labels, infer = False)  # Generates Batch-wise Adv Images
       # adv.requires_grad = False

        adv = adv.to(device)
        true_labels_adv = labels
        data = torch.cat((data, adv), 0)
        labels = torch.cat((labels, true_labels_adv))
        model.train()

        feats576, feats1024, outputs = model(data)
        loss_xent = cross_entropy_loss(outputs, labels)

        loss_prox_1024 = criterion_prox_1024(feats1024, labels)
        loss_prox_576 = criterion_prox_576(feats576, labels)

        loss_conprox_1024 = criterion_conprox_1024(feats1024, labels)
        loss_conprox_576 = criterion_conprox_576(feats576, labels)

        loss_prox_1024 *= args.weight_prox
        loss_prox_576 *= args.weight_prox

        loss_conprox_1024 *= args.weight_conprox
        loss_conprox_576 *= args.weight_conprox

        loss = loss_xent + loss_prox_1024 + loss_prox_576 - \
            loss_conprox_1024 - loss_conprox_576  # total loss
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

        if batch_idx % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})  XentLoss {:.6f} ({:.6f})  ProxLoss_1024 {:.6f} ({:.6f}) ProxLoss_576 {:.6f} ({:.6f}) \n ConProxLoss_1024 {:.6f} ({:.6f}) ConProxLoss_576 {:.6f} ({:.6f}) "
                  .format(batch_idx, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg,
                          prox_losses_1024.val, prox_losses_1024.avg, prox_losses_576.val, prox_losses_576.avg,
                          conprox_losses_1024.val, conprox_losses_1024.avg, conprox_losses_576.val,
                          conprox_losses_576.avg))
            
            writer.add_scalar('training loss',
                              losses.avg,
                              epoch * len(trainloader) + batch_idx)
            writer.add_scalar('training xent loss',
                              xent_losses.avg,
                              epoch * len(trainloader) + batch_idx)
            writer.add_scalar('training prox loss 1024',
                              prox_losses_1024.avg,
                              epoch * len(trainloader) + batch_idx)
            writer.add_scalar('training prox loss 576',
                              prox_losses_576.avg,
                              epoch * len(trainloader) + batch_idx)
            writer.add_scalar('training conprox loss 1024',
                              conprox_losses_1024.avg,
                              epoch * len(trainloader) + batch_idx)
            writer.add_scalar('training conprox loss 576',
                              conprox_losses_576.avg,
                              epoch * len(trainloader) + batch_idx)
            
            
            


def train_noCL(model, cross_entropy_loss,
               optimizer_model,
               trainloader, device, num_classes, epoch):

    #    model.train()
    losses = AverageMeter()
    att = Attack(model,cross_entropy_loss)

    # Batchwise training
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)
        model.eval()
        adv = att.pgd(data, labels, infer = False)  # Generates Batch-wise Adv Images
       # adv.requires_grad = False

        adv = adv.to(device)
        true_labels_adv = labels
        data = torch.cat((data, adv), 0)
        labels = torch.cat((labels, true_labels_adv))
        model.train()

        _, _, outputs = model(data)
        loss_xent = cross_entropy_loss(outputs, labels)

        loss = loss_xent
        optimizer_model.zero_grad()

        loss.backward()
        optimizer_model.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) )"
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))


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
