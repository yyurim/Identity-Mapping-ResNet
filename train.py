import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
import numpy as np
import time
import os

import model

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('current device: ', dev)

########################################
# data preparation: CIFAR10
########################################

batch_size = 8

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = model.IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

net.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()

        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Saving Model... ')
os.makedirs('./model',exist_ok=True)
torch.save(net.state_dict(), './model/final.pt')

print('Finished Training!')
