#!/usr/bin/env python3

import os, os.path, sys
import argparse
import importlib
import importlib.abc
import torch, torchvision
import torchvision.transforms as transforms

torch.seed()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if os.path.exists(project_dir) and os.path.isdir(project_dir) and os.path.isfile(module_filename):
        print("Found valid project in '{}'.".format(project_dir))
    else:
        print("Fatal: '{}' is not a valid project directory.".format(project_dir))
        raise FileNotFoundError

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)

    return project_module

def test_natural(net, test_loader, num_samples):
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            total = 0
            correct = 0
            for _ in range(num_samples):
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
    return valid

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", metavar="project-dir", nargs="?", default=os.getcwd(),
                        help="Path to the project directory to test.")
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Set batch size.")
    parser.add_argument("-s", "--num-samples", type=int, default=1,
                        help="Num samples for testing (required to test randomized networks).")

    args = parser.parse_args()
    project_module = load_project(args.project_dir)
    net = project_module.Net()
    net.to(device)
    net.load_for_testing(project_dir=args.project_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
    valid_loader = get_validation_loader(cifar, batch_size=args.batch_size)

    acc_nat = test_natural(net, valid_loader, num_samples = args.num_samples)
    print("Model nat accuracy (test): {}".format(acc_nat))

if __name__ == "__main__":
    main()
