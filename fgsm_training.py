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
import copy
from torch.autograd import Variable
from utils import AverageMeter, Logger, Proximity, Con_Proximity
from model import *



parser = argparse.ArgumentParser("Prototype Conformity Loss Implementation")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 50, 100],
                        help='Decrease learning rate at these epochs.')
parser.add_argument("--model-file", default=Net.model_file,
                     help="Name of the file used to load or to sore the model weights."\
                     "If the file exists, the weights will be load from it."\
                     "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                     "and the model weights will be stored in this file."\
                     "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
parser.add_argument('--lr_model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=200)
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

def attack(model, criterion, img, label, eps):
    adv = Variable(img,requires_grad=True)
    net = copy.deepcopy(model)
    out_adv = net(adv)
    loss = criterion(out_adv, label)
    
    net.zero_grad()
    loss.backward()

    adv.grad.sign_()
    adv = adv - eps*adv.grad
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data/',
                                             download=True,transform=transforms.Compose([transforms.ToTensor()]))
    indices = list(range(len(trainset)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, sampler=torch.utils.data.SubsetRandomSampler(indices[1024:]))

    testset = torchvision.datasets.CIFAR10(root='./data/',
                                            download=True,transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, sampler=torch.utils.data.SubsetRandomSampler(indices[:valid_size]))
    
# Loading the Model    
    model = Net(mode="small")
    model.load_state_dict(torch.load(args.model_file),strict=False)
    model.to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, momentum=0.9,weight_decay=5e-4)
    

    start_time = time.time()

    for epoch in range(args.max_epoch):
        
        adjust_learning_rate(optimizer_model, epoch)
        
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        if True:
            train_noCL(model, cross_entropy_loss, 
                  optimizer_model,trainloader, device, num_classes, epoch)

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")   #Tests after every 10 epochs
            acc, err , acc_adv , err_adv = test(model,cross_entropy_loss, testloader, device, num_classes, epoch)
            print("Accuracy (%): {:.5f}\t Error rate (%): {:.5f} | Accuracy adversarial(%): {:.5f}\t Error rate (%): {:.5f}".format(acc,err,acc_adv, err_adv))
                     
            torch.save(model.state_dict(),'Model_AdvTrain_FGSM.pth')
            
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train_noCL(model, cross_entropy_loss, 
              optimizer_model,
              trainloader, device, num_classes, epoch , alpha=0.5):
    
    #Batchwise training
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)
        model.eval()
        eps= 0.025
        adv = attack(model, cross_entropy_loss, data, labels, eps=eps) # Generates Batch-wise Adv Images
        adv.requires_grad= False
        
        adv= normalize(adv)
        adv= adv.to(device)
        #true_labels_adv= labels
        #data= torch.cat((data, adv),0)
        #labels= torch.cat((labels, true_labels_adv))
        model.train()
        
        outputs = model(data) 
        outputs_adv = model(adv)
        loss_x = cross_entropy_loss(outputs, labels)
        loss_xadv = cross_entropy_loss(outputs_adv, labels)
       
        
        loss = alpha*loss_x +(1-alpha)*loss_xadv
        optimizer_model.zero_grad()
        
        loss.backward()
        optimizer_model.step() 
        
   
        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f}" \
                  .format(batch_idx+1, len(trainloader), loss.item()))

def test(model,cross_entropy_loss, testloader, device, num_classes, epoch):
    model.eval()  
    correct, correct_adv, total = 0, 0, 0
    for param in model.parameters():
        param.requires_grad = False

    for data, labels in testloader:
        data, labels = data.to(device), labels.to(device)
        adv = attack(model, cross_entropy_loss, data, labels, eps=0.025) # Generates Batch-wise Adv Images
        adv.requires_grad= False
        adv= normalize(adv)
        adv= adv.to(device)
        
        outputs = model(data)
        outputs_adv = model(adv)
        predictions = outputs.data.max(1)[1]
        predictions_adv = outputs_adv.data.max(1)[1]
        total += labels.size(0)
        correct += (predictions == labels.data).sum()
        correct_adv += (predictions_adv == labels.data).sum()
            

    acc = correct * 100. / total
    err = 100. - acc
    acc_adv = correct_adv * 100. / total
    err_adv = 100. - acc_adv
    
    for param in model.parameters():
       param.requires_grad = True

    return acc, err ,acc_adv, err_adv

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
