# standard imports 

import torch # main pytorch library 
import torchvision # benchmark collection 
from torchvision import datasets, transforms # collection of data sets and functions 
import matplotlib.pyplot as plt # plotting and visualizing data 
import torch.nn as nn # neural network algorithm 
import torch.nn.functional as F 
import torch.optim as optim 


# separating train and test uploads of data 

train = datasets.MNIST('', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))

test = datasets.MNIST('', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle = True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle= True)

# class of neural net 

class Net(nn.Module): # few lines of code for neural network that can pass data with an output 

    def __init__(self): # initializes the self, super, and layers 

        super().__init__() # sub class inherits the attributes and methods of the nn.module 

        self.layer1 = nn.Linear(28*28, 64) # first layer takes in 28 x 28 images with outputs 64 connections 
        self.layer2 = nn.Linear(64, 64) # second layer takes the output from the previous layer

        self.layer3 = nn.Linear(64,64) # third layer is meant to repeat 

        self.layer4 = nn.Linear(64, 10) # 10 neurons for 10 classes  

    def forward(self, x): # needs to be called forward just inputs the inputs to feed forward through activation function

        x = F.relu(self.layer1(x))

        x = F.relu(self.layer2(x))

        x = F.relu(self.layer3(x)) 

        x = self.layer4(x) 

        return F.log_softmax(x, dim=1) 

# intializing

net = Net() # calls the class Net's function operations 
loss_function = nn.CrossEntropyLoss() # calculates how far fof our classifications are 
optimizer = optim.Adam(net.parameters(), lr=0.001) # tweaking the parameters and adjusting the weights , learning rate of 0.001 

# running the net

for epoch in range(3): # 3 full passes over the data

    for data in trainset:  # `data` is a batch of data

        X, y = data  # X is the batch of features, y is the batch of targets.

        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.

        output = net(X.view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)

        loss = F.nll_loss(output, y)  # calc and grab the loss value

        loss.backward()  # apply this loss backwards thru the network's parameters
        
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
print(loss) 

# accuracy with epoch and batch 

correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1,784))
        #print(output)
        for idx, i in enumerate(output):
            #print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

# debugging - kernel crash 

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'