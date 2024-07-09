import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from utils import *
from nets import *
import pdb
from torch.utils.data import DataLoader
from torch import optim
import datetime
import random
import time

class CustomCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_labels=[0, 1], n=5):
        self.cifar_data = datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
        
        # Filter data to include only specified target labels
        self.data = []
        self.targets = []
        for idx, (data, target) in enumerate(self.cifar_data):
            if target in target_labels:
                self.data.append(data)
                self.targets.append(target)
            # Load only 1/n of the data
            if len(self.data) >= len(self.cifar_data)//n:
                break
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create custom datasets and data loaders
num_clients = 100
train_dataset = CustomCIFAR10(root='data', train=True, transform=transform_train, target_labels=[9, 0], n=100)
test_dataset = CustomCIFAR10(root='data', train=False, transform=transform_test, target_labels=[9, 0], n=100)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

client_model = SimpleUserNet().to(device).train()
server_model = SimpleServerNet().to(device).train()

learningRate = 1e-3
eps = 1e-3
AMSGrad = True
optimizer = optim.Adam([
    {'params': client_model.parameters()},
    {'params': server_model.parameters()}
], lr=learningRate, eps=eps, amsgrad=AMSGrad)

epochs = 100
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    correct = 0
    total = 0
    train_loss = 0
    avg_train_acc = []

    # Train the user model on the client's dataset
    start_time = time.time()
    for _, (images, labels) in enumerate(train_loader):
        
        images, labels = images.to(device), labels.to(device)
        pdb.set_trace()
        # Forward pass through the server model
        front_output = server_model(images)
        
        # Forward pass through the user model
        user_output = client_model(front_output)

        # Calculate the loss and perform backpropagation
        optimizer.zero_grad()
        loss = criterion(user_output, labels)
        loss.backward()

        optimizer.step()

        _, pred = torch.max(user_output, 1)
        correct += torch.sum(np.squeeze(pred.eq(labels.data.view_as(pred))))
        train_loss += loss.item()
        total += labels.shape[0]
    end_time = time.time()
    print(f'One batch training time: {end_time-start_time}')

    print('\n### Train Loss: %.4f,  Train Acc.: %.4f ###' % (train_loss/total , correct/total))
    avg_train_acc.append(correct/total)

    with torch.no_grad():
        test_acc = []
        correct = 0
        total = 0
        test_loss = 0
        server_model = server_model.eval()
        client_model = client_model.eval()

        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            front_ouput = server_model(images)
            output = client_model(front_ouput)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss.item()

        server_model.train()
        client_model.train()
        print('### Avg. Test Acc. : %4f ###' % ( correct/total ))