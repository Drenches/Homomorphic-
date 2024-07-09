import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import tenseal as ts
from utils import *
from fedlab.utils.dataset import MNISTPartitioner
from torch.utils.data import SubsetRandomSampler
import copy
from nets import *
from cifar10_data_loader import *
import random
import pdb
import datetime

## Load data and model
gpu = torch.cuda.is_available()

batch_size = 512
num_clients = 100
num_rounds = 500
num_epoches = 5
major_classes_num = 9
sample_rate = 0.1
data_names = ['mnist', 'cifar10']
dataset = data_names[1]
print(dataset)

if dataset == 'mnist':
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    # Split the dataset into multiple clients

    # client_datasets_part = MNISTPartitioner(
    #     train_data.targets,
    #     num_clients=num_clients,
    #     partition="noniid-#label",
    #     major_classes_num=major_classes_num
    # )
    client_datasets_part = MNISTPartitioner(
        train_data.targets,
        num_clients=num_clients,
        partition="iid",
    )
    # test_part = MNISTPartitioner(
    #     test_data.targets,
    #     num_clients=num_clients,
    #     partition="noniid-#label",
    #     major_classes_num=major_classes_num
    # )
    test_part = MNISTPartitioner(
        test_data.targets,
        num_clients=num_clients,
        partition="iid",)
    model = LeNet5().cuda()

elif dataset == 'cifar10':
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, DATA_CLASS = load_partition_data_cifar10('data', num_clients, 5, batch_size)
    # model = CombinedNet().cuda()
    # model = SimpleConvNet().cuda()
    model = CIFAR10CNN().cuda()

# Define FL process 
criterion = torch.nn.CrossEntropyLoss()
def train(model, train_loader, num_epoches = 3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(num_epoches):
        for batch_idx, (data, target) in enumerate(train_loader):
            if dataset == 'cifar10':
                data, target = data.permute(0, 3, 1, 2).cuda(), target.cuda()
            else:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_acc(output, target)
    return model

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            if dataset == 'cifar10':
                data, target = data.permute(0, 3, 1, 2).cuda(), target.cuda()
            else:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += torch.sum(criterion(output, target)).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            counter += target.shape[0]
    test_loss /= counter
    # len(test_loader.dataset)
    accuracy = correct / counter
    return test_loss, accuracy

def federated_train(model, sample_rate=0.2, num_rounds=5, num_epoches = 3, num_clients=10):
    # global_weights = copy.deepcopy(model.state_dict())
    client_list = [i for i in range(num_clients)]
    for round in range(num_rounds):
        print("Round:", round)
        local_models = []
        random_selected_clients_list = random.sample(client_list, int(len(client_list)*sample_rate))
        for client_id in random_selected_clients_list:
            # Evaluate global model on local test data every 10 rounds
            if round%2  == 0:
                if dataset == "mnist":
                    test_loader = torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(client_datasets_part[client_id]), batch_size=2048)
                elif dataset == 'cifar10':
                    test_loader = test_data_local_dict[client_id]
                test_loss, accuracy = test(model, test_loader)
                print(f'Client {client_id + 1}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.5f}')

            # Get local model and train on client data
            local_model = copy.deepcopy(model)
            if dataset == "mnist":
                train_loader = torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(client_datasets_part[client_id]), batch_size=batch_size)
            elif dataset == "cifar10":
                train_loader = train_data_local_dict[client_id]
            local_model = train(local_model, train_loader, num_epoches)
            local_models.append(local_model)

        # Average the local models to get the updated global model weights
        model = aggregation(local_models)

        
federated_train(model, sample_rate, num_rounds=num_rounds, num_epoches = num_epoches, num_clients=num_clients)
        


