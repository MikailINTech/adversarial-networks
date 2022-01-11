import numpy as np
import torch
import matplotlib.pyplot as plt

'''
This file is used to make visual results
'''

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']



def compare(img1, img2, legends = None):
    '''
    Takes images right before feeding them to the network, aka torch tensor
    legends : should be an array or tuple of size 2 with the predicted number
    '''
    f, ax = plt.subplots(1,2)
    #f.get_xaxis().set_visible(False)
    #f.get_yaxis().set_visible(False)
    ax[0].imshow(img1.cpu().detach().numpy().T)
    ax[1].imshow(img2.cpu().detach().numpy().T)
    if legends:
        print(legends)
        ax[0].title.set_text(f"Predicted {classes[legends[0]]}")
        ax[1].title.set_text(f"Predicted {classes[legends[1]]}")
    plt.show()