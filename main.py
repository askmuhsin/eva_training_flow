import sys
sys.path.append('./')

from utils import (
    setup, data, viz
)
from utils.training import train
from utils.testing import test

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

setup.set_seed()
cuda = setup.is_cuda()
device = setup.get_device()

print(f"[INFO] Loading Data")
train_loader = data.CIFAR10_dataset(
    train=True, cuda=cuda
).get_loader()
test_loader = data.CIFAR10_dataset(
    train=False, cuda=cuda
).get_loader()    
    
def train_model(epochs, model, lr=0.01):
    net = model.to(device)
    viz.show_model_summary(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=lr,
        momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    EPOCHS = epochs

    print(f"[INFO] Begin training for {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        train_batch_loss, train_batch_acc= train(
            net, device, 
            train_loader, optimizer, criterion, epoch,
        )
        train_loss = np.mean(train_batch_loss)
        train_acc = np.mean(train_batch_acc)
        test_loss, test_acc = test(
            net, device,
            test_loader, criterion, epoch,
        )
        scheduler.step()
    
    return net
