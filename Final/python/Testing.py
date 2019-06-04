import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
import random

from sklearn.metrics import confusion_matrix

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
    def __init__(self,root='/opt/lampp/htdocs/new/python/data/test/', transform=None):
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



# Testing -------------------------------------------------

counter = 1
accuracy = []

train_dataset = MyDataset('/opt/lampp/htdocs/new/python/data/test/', transforms.ToTensor())
# image, target = train_dataset.__getitem__(0)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

testLoss = 0.0
correct = 0

for _, (image,label) in enumerate(train_loader):
    image = Variable(image)
    label = Variable(label)
    
    model = torch.load('/opt/lampp/htdocs/new/python/data/model.pth')
    # model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    predictedOutput = model(image)
    # print(predictedOutput)
    testLoss += loss_function(predictedOutput,label)

    predictedDigit = predictedOutput.data.max(1)[1]
    print(predictedDigit,' ',label)
    correct += predictedDigit.eq(label.data).sum()

    # print('Loss = ',loss/54)

correct = correct.cpu()
print(counter, 'correct : ', np.array(correct), '/', 
    len(train_loader.dataset), '-> Accuracy : ', 
    100 * float(correct)/len(train_loader.dataset))
counter += 1
accuracy.append(100*correct/len(train_loader.dataset))
del train_dataset, train_loader

# ----------------------------------------------------