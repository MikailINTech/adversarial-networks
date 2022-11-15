import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import matplotlib_imshow, plot_classes_preds



import DDNN


now = datetime.now()
logdir = "runs/" + now.strftime("%Y%m%d-%H") + "/"

model_file = "models/default_model_adv2.pth"
denoiser_file = "./models/denoise_DnCNN.pth"

'''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model-file", default=model_file,
                    help="Name of the file used to load or to sore the model weights."
                    "If the file exists, the weights will be load from it."
                    "If the file doesn't exists, or if --force-train is set, training will be performed, "
                    "and the model weights will be stored in this file."
                    "Warning: "+model_file+" will be used for testing (see load_for_testing()).")

parser.add_argument('-f', '--force-train', action="store_true",
                    help="Force training even if model file already exists"
                         "Warning: previous model file will be erased!).")

parser.add_argument('-e', '--num-epochs', type=int, default=10,
                    help="Set the number of epochs during training")

parser.add_argument('-bs', '--batch-size', type=int, default=64)
parser.add_argument('-vs', '--valid-size', type=int, default=1024)

args = parser.parse_args()

writer = SummaryWriter(logdir)

torch.manual_seed(24)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Now using  {}".format("GPU" if use_cuda else "CPU"))

valid_size = args.valid_size
batch_size = args.batch_size
freeze = False


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):

    def __init__(self, mode="small", num_classes=1000, width_mult=1., pretrain=False, adv_output=False):
        super(Net, self).__init__()
        # setting of inverted residual blocks
        assert mode in ['large', 'small']
        if mode == "small":
            self.cfgs = [
                # k, t, c, SE, HS, s
                [3,    1,  16, 1, 0, 2],
                [3,  4.5,  24, 0, 0, 2],
                [3, 3.67,  24, 0, 0, 1],
                [5,    4,  40, 1, 1, 2],
                [5,    6,  40, 1, 1, 1],
                [5,    6,  40, 1, 1, 1],
                [5,    3,  48, 1, 1, 1],
                [5,    3,  48, 1, 1, 1],
                [5,    6,  96, 1, 1, 2],
                [5,    6,  96, 1, 1, 1],
                [5,    6,  96, 1, 1, 1],
            ]
        else:
            self.cfgs = [
                # k, t, c, SE, HS, s
                [3,   1,  16, 0, 0, 1],
                [3,   4,  24, 0, 0, 2],
                [3,   3,  24, 0, 0, 1],
                [5,   3,  40, 1, 0, 2],
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1],
                [3,   6,  80, 0, 1, 2],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1],
                [5,   6, 160, 1, 1, 2],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]
            ]
            
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size,
                          output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(
            output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.exp_size = exp_size
        self.output_channel = output_channel
        self.classifier = nn.Sequential(
           nn.Linear(self.exp_size, self.output_channel),
           h_swish(),
           nn.Dropout(0.2),
           nn.Linear(self.output_channel, num_classes),
           )


        self._initialize_weights()
        
        if pretrain:
            if mode == "small":
                self.load_state_dict(torch.load(
                    "./models/mobilenetv3-small-55df8e1f.pth"))
            else:
                self.load_state_dict(torch.load(
                    "./models/mobilenetv3-large-1cd25616.pth"))

        self.adv_output = adv_output
        self.int_classifier = nn.Linear(self.exp_size, self.output_channel)
        self.swish = h_swish()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.output_channel, 10)
        # self.transforms_init()

    def forward(self, x, denoise=False):

        if denoise:
            x = x + 0.1 * torch.randn(*x.shape).to(device)
            x = self.denoiser(x)
        x = self.features(x)
        x = self.conv(x)

        x = self.avgpool(x)
        z = x.view(x.size(0), -1)
        x = self.int_classifier(z)
        y = self.swish(x)
        y = self.dropout(y)
        y = self.classifier(y)

        if self.adv_output:
            return z, x, y
        else:
            return y

    def transforms_init(self):
        self.denoiser = DDNN.DnCNN()
        self.denoiser.load(self.denoiser_file)
        print("Denoiser Loaded")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(
            model_file, map_location=torch.device(device)), strict=True)

    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.

           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''
        self.load(os.path.join(project_dir, self.model_file))


if freeze:
    for param in Net.parameters():
        param.requires_grad = False
    for param in Net.classifier.parameters():
        param.requires_grad = True


def train_model(net, train_loader, pth_filename, num_epochs, val_loader=None):
    print("Starting training")
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200)

    for epoch in range(num_epochs):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            _,_,outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 500 == 0:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss /
                      (batch_idx+1), 100.*correct/total, correct, total))
                print(predicted, targets)
                # ...log the running loss
                writer.add_scalar('training loss',
                                  train_loss/(batch_idx+1),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Accuracy', 100.*correct/total,
                                  epoch * len(train_loader) + batch_idx)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(
                                      net, inputs, targets, classes),
                                  global_step=epoch * len(train_loader) + batch_idx)

        scheduler.step()

        if val_loader:
            acc = test_natural(net, val_loader)
            print("Model natural accuracy (valid): {}".format(acc))

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))


def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            _,_,outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def get_train_loader(dataset, batch_size, valid_size):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler, batch_size=batch_size)

    return train


def get_validation_loader(dataset, batch_size, valid_size):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(
        dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid


def main():

    # Create model and move it to whatever device is available (gpu/cpu)
    net = Net(pretrain=True,adv_output=True)
    net.to(device)

    cifar = torchvision.datasets.CIFAR10(
        './data/', download=True, transform=transforms.ToTensor())
    valid_loader = get_validation_loader(cifar, batch_size, valid_size)

    # get some random training images
    dataiter = iter(valid_loader)
    images, labels = next(dataiter)
    images = images.to(device)

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('cifar10_images', img_grid)
    writer.add_graph(net, images)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    # features = images.view(-1, 3*32 * 32)
    # writer.add_embedding(features,
    #                      metadata=class_labels,
    #                      label_img=images)

    # Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        cifar = torchvision.datasets.CIFAR10(
            './data/', download=True, transform=transforms.ToTensor())
        train_loader = get_train_loader(
            cifar, batch_size=batch_size, valid_size=valid_size)
        train_model(net, train_loader, args.model_file,
                    args.num_epochs, valid_loader)
        print("Model save to '{}'.".format(args.model_file))

    # log embeddings
    _,features,_ = net(images)
    writer.add_embedding(features,
                         metadata=class_labels,
                         label_img=images)

    # Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

    if args.model_file != model_file:
        print("Warning: '{0}' is not the default model file, "
              "it will not be the one used for testing your project. "
              "If this is your best model, "
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, model_file))

    writer.close()


if __name__ == "__main__":
    main()
