import torch.nn as nn
import torch.nn.functional as F
import os

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt

def imshow(img):
    # helper function to un-normalize and display an image
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model_file = "./models/denoise_DnCNN5epoch.pth"


# define the NN architecture
class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        return y - residual
        
    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    
# initialize the NN
model = DnCNN()
model.cuda()


def train(model, nepochs, train_loader, lr, nf):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for epoch in range(1, nepochs+1):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # no need to flatten images
            images, _ = data
            
            ## add random noise to the input images
            noisy_imgs = images + nf * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)
                    
            optimizer.zero_grad()
            outputs = model(noisy_imgs.cuda())
            loss = criterion(outputs, images.cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))

def test(model,test_loader,nf, batch_size):
        
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # add noise to the test images
    noisy_imgs = images + nf * torch.randn(*images.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)

    plt.figure(figsize=(20, 6))
    n = 10
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        imshow(images[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(3, n, i +1 + n)
        imshow(noisy_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display denoised image
        ax = plt.subplot(3, n, i +1 + n + n)
        output = model(noisy_imgs.cuda())
        output = output.view(batch_size, 3, 32, 32)
        output = output.detach().cpu()
        imshow(output[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.figtext(0.5,0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.figtext(0.5,0.65, "NOISY IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.figtext(0.5,0.35, " DENOISED RECONSTRUCTED IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.subplots_adjust(hspace = 0.5 )
    plt.show()


def usableM():
    model.load(model_file)
    return(model)

def main():
    # convert data to a normalized torch.FloatTensor
    transform = transforms.ToTensor()
    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                download=True, transform=transform)

    noise_factor=0.1
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


    valid_size = 1024
    batch_size = 32
    epoch = 5
    learning_rate = 0.001
    save = True
    training = False

    if training:
    #### Model training (if necessary)
        print("Training model")

        train(model, epoch, train_loader, learning_rate, noise_factor)

        if save:
            model.save(model_file) 
            print("Model Saved")
    
    model.load(model_file)
    test(model, test_loader, noise_factor, batch_size)

if __name__ == "__main__":
    main()
