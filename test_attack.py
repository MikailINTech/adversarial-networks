from attacks import Attack
from model import Net, get_validation_loader
import argparse

import torch, torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import showres

from utils import plot_classes_preds

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model_file = "models/base_model.pth"

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser()
parser.add_argument("--model-file", default=model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+model_file+" will be used for testing (see load_for_testing()).")
parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")

parser.add_argument('-bs', '--batch-size', type=int, default=64)
parser.add_argument('-vs', '--valid-size', type=int, default=1024)

args = parser.parse_args()
    
valid_size = args.valid_size
batch_size = args.batch_size
show = True

def compare_prediction(net,test_loader,att):
    correct = 0
    correct_adv = 0
    total = 0
    for i,data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        #new_labels = torch.randint(0,9,labels.shape).to(device)
        #new_labels = torch.zeros_like(labels).to(device)
        new_labels = 9-labels
        img_adv,outputs_adv,outputs= att.pgd(images,new_labels,True,0.025)
        _, predicted = torch.max(outputs.data, 1)
        _, predicted_adv = torch.max(outputs_adv.data, 1)
        if show and i in range(0,5):
            showres.compare(images[0],img_adv[0],(predicted[0],predicted_adv[0]))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        correct_adv += (predicted_adv == labels).sum().item()
        plot_classes_preds(net, img_adv, labels, classes)

    return 100 * correct / total, 100 * correct_adv / total

def main():

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net(mode="small",pretrain=True,adv_output=True)
    net.load_state_dict(torch.load(args.model_file),strict=False)
    net.to(device)
    
    #summary(net, (3, 32, 32), 8)
    
    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))
    
    criterion = nn.CrossEntropyLoss()
    
    
    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar,batch_size,valid_size)
    
    pgd = Attack(net,criterion)
    
    acc , acc_adv = compare_prediction(net, valid_loader,pgd)
    print("Model natural accuracy (valid): {} vs accuracy after attack : {}".format(acc,acc_adv))
    
    if args.model_file != model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, model_file))

if __name__ == "__main__":
    main()