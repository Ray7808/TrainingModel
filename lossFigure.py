import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(n, path):
    # Genereate the loss figure

    y = []
    for i in range(0,n):
        #enc = np.load(path+'\UNet_100\loss\epoch_{}.npy'.format(i))
        enc = np.load(path+'\loss\epoch_{}.npy'.format(i))
        # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
        tempy = list(enc)
        y += tempy
    x = range(0,len(y))
    plt.plot(x, y, '.-')
    plt_title = 'BATCH_SIZE = 10; LEARNING_RATE:1e-3'
    plt.title(plt_title)
    plt.xlabel('per 1 times')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    #Here need to revise the number of Epoch
    Epoch=100
    path = os.getcwd()
    plot_loss(Epoch, path)