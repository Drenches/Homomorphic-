import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from utils import *
from nets import *
from fedlab.utils.dataset import MNISTPartitioner,CIFAR10Partitioner
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler
import pdb

num_clients = 10
num_classes = 10
seed = 2021
transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
# datasets_part  = CIFAR10Partitioner(train_data.targets,
#                                     num_clients,
#                                     balance=True,
#                                     partition="iid",
#                                     seed=seed)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024,
                                          shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(datasets_part[0]), batch_size=512)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024,
#                                           shuffle=True)

net1 = ServerNetCIFAR10()
net2 = UserNetCIFAR10()

criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

distribution_list = []
net1.cuda()
net2.cuda()
criterion.cuda()
for epoch in range(200):
    correct = 0
    total = 0
    train_loss = 0
    for images, labels in train_loader:
        images, labels = next(iter(train_loader))
        images, labels = images.cuda(), labels.cuda()
        distribution_list += torch.bincount(labels.flatten(), minlength=num_classes)
        
        front_ouput = net1(images)
        output = net2(front_ouput)
        loss = criterion(output, labels)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss.backward()

        optimizer1.step()
        optimizer2.step()

        _, pred = torch.max(output, 1)
        correct += torch.sum(np.squeeze(pred.eq(labels.data.view_as(pred))))
        train_loss += loss.item()
        total += labels.shape[0]
    print('Epoch: %d, Train Loss: %.3f,  Train Acc.: %.3f' % (epoch, train_loss/total , correct/total))
    

    correct = 0
    total = 0
    net1.eval()
    net2.eval()
    with torch.no_grad():
        # distribution = TestSampGen(test_data, distribution_list)
        # sampler = WeightedRandomSampler(weights=distribution[0].tolist(), replacement=True, num_samples=len(test_data)//num_clients)
        # test_loader = torch.utils.data.DataLoader(test_data, sampler=sampler, batch_size=2048)
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            front_ouput = net1(images)
            output = net2(front_ouput)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    net1.train()
    net2.train()
pdb.set_trace()
