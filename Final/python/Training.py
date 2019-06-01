import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

torch.manual_seed(0)


# CNN model class ----------------------------------------

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        
        self.fc1 = nn.Linear(8*9*20,10)
        self.fc2 = nn.Linear(10,2)
        
    def forward(self,image):
        image = self.cnn1(image)
        image = self.maxpool1(image)
        image = self.relu1(image)
        
        image = self.cnn2(image)
        image = self.maxpool2(image)
        image = self.relu2(image)
        
        image = self.cnn3(image)
        image = self.maxpool3(image)
        image = self.relu3(image)
        
        image = image.view(image.size(0), -1)
        # image = self.dropOut(image)
        image = F.relu(self.fc1(image))
        image = self.fc2(image)
        
        return image

# -----------------------------------------------------------------



# Loading data sets ----------------------------------------------

class MyDataset(Dataset):
    def __init__(self,root='/opt/lampp/htdocs/new/python/data/train/', transform=None):
        self.root = root
        self.files = os.listdir(root)
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        self.file_name = self.files[index]
        self.file_path = os.path.join(self.root,self.file_name)
        self.target = 0
        if(self.file_name[:2] == 'cf'):
            self.target = 0
        else:
            self.target = 1
        image = Image.open(self.file_path)
        scaler = StandardScaler()
        image = scaler.fit_transform(image)
        image = self.transform(image)
        self.target = torch.tensor(self.target)

        return (image, self.target)

# ------------------------------------------------------------------


model = CNN_Model()
optimizer = optim.SGD(model.parameters(),lr = 0.001, momentum = 0.9)
loss_function = nn.CrossEntropyLoss()

def plot_graph(loss_list):
    epoch_list = list(i for i in range(len(loss_list)))
    plt.plot(epoch_list,loss_list, '-')


# Traning ---------------------------------------------

epochs = 50
loss_list = []   
    
model = CNN_Model()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

# model = torch.load('/opt/lampp/htdocs/new/python/model_cnn_gpu_1(1, 30, 110).pth')
# model = model.to('cpu')
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# # optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss_function = nn.CrossEntropyLoss()


for epoch in range(epochs):
    epoch_loss = 0.0
    total_dataset_length = 0
        
    for j in [1, 3]:
        
        train_dataset = MyDataset('/opt/lampp/htdocs/new/python/data/train/', transforms.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

        for _,(image,label) in (enumerate(train_loader)):

            image = Variable(image)
            label = Variable(label)

            optimizer.zero_grad()
            model = model.double() # This has been used because Torch by default considers float Tensors but our tensors are double so we change the model to double tensors
            output = model(image)   # output gives the probability distribution over the variables
            loss = loss_function(output, label) # cross entropy then finds the loss
            # final_output = output.data.max(1)[1]
            # correct += final_output.eq(label.data).sum()
            loss.backward()
            optimizer.step()

            epoch_loss += output.shape[0] * loss.item()
            total_dataset_length += len(train_dataset)
                
    print('loss epoch {} : '.format(epoch+1), epoch_loss/len(train_dataset), '\n')
    # print('loss epoch {} : '.format(epoch+1),epoch_loss/total_dataset_length)
    loss_list.append(epoch_loss/len(train_dataset))
    # loss_list.append(epoch_loss/total_dataset_length)
    torch.save(model, './python/data/model.pth')
    # plot_graph(loss_list)

    del train_dataset, train_loader
    
# ----------------------------------------------------------------




# print('Hello Shivam')
