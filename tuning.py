import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from utils import *
from nets import *
from cifar10_data_loader import *
import pdb
from torch.utils.data import DataLoader
import datetime
# from fedlab.utils.dataset import MNISTPartitioner,CIFAR10Partitioner
# from torch.utils.data import SubsetRandomSampler
# from torch.utils.data import WeightedRandomSampler
import random


def tuning(model1, model2, train_loader, test_loader, num_epoches = 200):
    model1.train()
    model2.train()
    learningRate = 1e-3
    eps = 1e-3
    AMSGrad = True
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learningRate, eps=eps, amsgrad= AMSGrad)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learningRate, eps=eps, amsgrad=AMSGrad)
    for _ in range(num_epoches):
        for batch_idx, (data, target) in enumerate(train_loader):
            # if dataset == 'cifar10':
            data, target = data.permute(0, 3, 1, 2).cuda(), target.cuda()
            # else:
                # data, target = data.cuda(), target.cuda()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            x = model1(data)
            output = model2(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            train_acc(output, target)
        
        model1.eval()
        model2.eval()
        test_loss = 0
        correct = 0
        counter = 0
        with torch.no_grad():
            for data, target in test_loader:
                # if dataset == 'cifar10':
                data, target = data.permute(0, 3, 1, 2).cuda(), target.cuda()
                # else:
                    # data, target = data.cuda(), target.cuda()
                x = model1(data)
                output = model2(x)
                test_loss += torch.sum(criterion(output, target)).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                counter += target.shape[0]
        test_loss /= counter
        accuracy = correct / counter
        print(f'Test loss: {test_loss:.4f}, Accuracy: {accuracy:.5f}')

num_clients = 10
client_model_list = [UserNetCIFAR10().cuda().train() for _ in range(num_clients)]
globe_server_model = ServerNetCIFAR10().cuda().train()

root_path = '/home/dev/workspace/Homomorphic-HalfFed/saved_weights/50/'
globe_server_model.load_state_dict(torch.load(root_path+str(0)))
# for i in range(1, num_clients):
#     client_model_list[i].load_state_dict(torch.load(root_path+str(i)))

batch_size = 128
train_data_num, test_data_num, train_data_global, test_data_global, \
train_data_local_num_dict, train_data_local_dict, test_data_local_dict, DATA_CLASS = load_partition_data_cifar10('data', num_clients, 10, batch_size)


cliend_id = 8
tuning(globe_server_model, client_model_list[cliend_id], train_data_local_dict[cliend_id], test_data_local_dict[cliend_id])