import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from utils import *
from nets import *
from cifar10_data_loader import *
import pdb
import datetime
from fedlab.utils.dataset import MNISTPartitioner,CIFAR10Partitioner
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler
import random

## Load data and model
gpu = torch.cuda.is_available()
# gpu = False
# entire_model = SimpleConvNet()
batch_size = 16
major_classes_num=3
num_clients = 10
seed = 2021
data_name = 'MNIST'

transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if data_name=='MNIST':
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    # Split the dataset into multiple clients
    # This part is implenmented by FedLab: https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html
    # major class division
    client_datasets_part = MNISTPartitioner(
        train_data.targets,
        num_clients=num_clients,
        partition="noniid-#label",
        major_classes_num=major_classes_num,
        seed = seed
    )

    # dir alpha class division
    # client_datasets_part = MNISTPartitioner(train_data.targets, 
    #                                     num_clients=num_clients,
    #                                     partition="noniid-labeldir", 
    #                                     dir_alpha=0.5,
    #                                     seed=seed)
    
    #load server model
    globe_server_model = ServerNetMNIST().cuda()
    # server_model_list = [copy.deepcopy(server_model_init) for _ in range(num_clients)]


    
num_classes = torch.bincount(torch.Tensor(test_data.targets).int()).shape[0]

# User model list
if data_name=='MNIST':
    client_model_list = [UserNetMNIST(output=num_classes).cuda().train() for _ in range(num_clients)]
if data_name=='CIFAR10':
    client_model_list = [UserNetCIFAR10(output=num_classes).cuda().train() for _ in range(num_clients)]

# client_datasets = torch.utils.data.random_split(train_data, [len(train_data)//num_clients]*num_clients)
client_dataloaders = [torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(indices=client_datasets_part[i]), batch_size=batch_size) for i in range(num_clients)]

# Define the loss function and the server's optimizer
criterion = torch.nn.CrossEntropyLoss()
# client_optimizer_list = [torch.optim.SGD(client_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) for client_model in client_model_list]
client_optimizer_list = [torch.optim.Adam(client_model.parameters(), lr=0.01) for client_model in client_model_list]

# Muti-round training
def train(round = 50000, epoches = 5, p=0.2):
    server_model_list = [copy.deepcopy(globe_server_model) for _ in range(num_clients)]
    server_optimizer_list = [torch.optim.SGD(server_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) for server_model in server_model_list]
    client_list = [i for i in range(num_clients)]

    # record labels distribution
    distribution_list = [torch.zeros(num_classes).cuda() for _ in range(num_clients)]

    for i in range(round):
        print('round:', i)
        # randomly select a subset of clients
        random_selected_clients_list = random.sample(client_list, int(len(client_list)*p))

        for client_id in random_selected_clients_list:

            # Train multiple epoches for selected clients
            # Load client's dataset
            client_dataloader = client_dataloaders[client_id]
            client_optimizer = client_optimizer_list[client_id]
            client_model = client_model_list[client_id].train()

            # Load server model corresonding for each specific client
            server_model = server_model_list[client_id].train()
            server_optimizer =server_optimizer_list[client_id]
            
            correct = 0
            total = 0
            train_loss = 0
            for epoch in range(epoches):

                client_optimizer.zero_grad()
                server_optimizer.zero_grad()

                # Train the user model on the client's dataset
                images, labels = next(iter(client_dataloader))
                images, labels = images.cuda(), labels.cuda()
                distribution_list[client_id] += torch.bincount(labels.flatten(), minlength=num_classes)

                # Forward pass through the server model
                front_output = server_model(images)
                pdb.set_trace()
                
                # Forward pass through the user model
                user_output = client_model(front_output)
                loss = criterion(user_output, labels)

                # Calculate the loss and perform backpropagation

                loss.backward()

                client_optimizer.step()
                server_optimizer.step()
                

                _, pred = torch.max(user_output, 1)
                correct += torch.sum(np.squeeze(pred.eq(labels.data.view_as(pred))))
                train_loss += loss.item()
                total += labels.shape[0]

            # Calculate the train accuracy of each client
            print('Client ', client_id)
            print('Train Loss: %.4f,  Train Acc.: %.4f' % (train_loss/total , correct/total))
        

        # Aggregate and update the server model
        updated_server_models = [server_model_list[i] for i in random_selected_clients_list]
        new_global_model = aggregation(updated_server_models)
        server_model_list = [copy.deepcopy(new_global_model) for _ in range(num_clients)]
        # new_global_client_model = aggregation(client_model_list)
        # client_model_list = [new_global_client_model]*num_clients
        # print('server model aggregated')


        # test(distribution_list=distribution_list.copy(), server_model_list= server_model_list)
        if i%50==0 and i!=0:
            with torch.no_grad():
                distribution = TestSampGen(test_data, distribution_list.copy())
                for client_id in range(num_clients):
                    sampler = WeightedRandomSampler(weights=distribution[client_id].tolist(), replacement=True, num_samples=len(test_data)//num_clients)
                    test_loader = torch.utils.data.DataLoader(test_data, sampler=sampler, batch_size=2048)
                    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2048)
                    correct = 0
                    total = 0
                    test_loss = 0
                    for images, labels in test_loader:
                        images, labels = images.cuda(), labels.cuda()
                        server_model = server_model_list[client_id].eval()
                        client_model = client_model_list[client_id].eval()
                        front_ouput = server_model(images)
                        output = client_model(front_ouput)
                        loss = criterion(output, labels)
                        _, predicted = torch.max(output.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        test_loss += loss.item()
                    print('Client ', client_id)
                    print('Test Loss: %.4f,  Test Acc.: %.4f' % (train_loss/total , correct/total))
                    server_model.train()
                    client_model.train()

        
        # pdb.set_trace()
        # if i%10==0:  
            
        
        # if i%10==0:  
        #     test_acc = test(distribution_list=distribution_list.copy())
        #     if abs(test_acc-acc_counter/num_clients)>0.2:
        #         server_stop_marker = 1
        # if server_stop_marker: 
        #     server_model.eval()
        #     print('Server part stops updating')
        # else:
        #     server_model.train()
        #     server_optimizer.step()
        #     print('Server part updated')
        
        # # Alternate update
        # if i%10==0:  
        #     test_acc = test(distribution_list=distribution_list.copy())
        #     if abs(test_acc-acc_counter/num_clients)>0.2:
        #         server_stop_marker = 1
        # if server_stop_marker: 
        #     server_model.eval()
        #     print('Server part stops updating')
        # else:
        #     server_optimizer.step()
        #     print('Server part updated')
        
        

# def test(test_batch_size = 2048, distribution_list=None, server_model_list=None):
#     distribution = TestSampGen(test_data, distribution_list)
#     total_correct = 0
#     total_vol = 0
#     for client_id in range(num_clients):
#         client_model = client_model_list[client_id].eval()
#         server_model = server_model_list[client_id].eval()
#         # test_loader = torch.utils.data.DataLoader(test_data, sampler=SubsetRandomSampler(test_part[client_id]), batch_size=test_batch_size)
#         sampler = WeightedRandomSampler(weights=distribution[client_id].tolist(), replacement=True, num_samples=len(test_data)//num_clients)
#         test_loader = torch.utils.data.DataLoader(test_data, sampler=sampler, batch_size=test_batch_size)
#         test_loss = 0.0
#         class_correct = list(0. for i in range(10))
#         class_total = list(0. for i in range(10))
#         for images, labels in test_loader:
#             images, labels = images.cuda(), labels.cuda()
#             server_output = server_model(images)
#             user_output = client_model(server_output)
#             loss = criterion(user_output, labels)
#             test_loss += loss.item()
#             _, pred = torch.max(user_output, 1)
#             correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
#             for i in range(len(labels)):
#                 label = labels.data[i]
#                 class_correct[label] += correct[i].item()
#                 class_total[label] += 1
#             total_vol += labels.shape[0]
        
#         total_correct += torch.sum(correct).item()
#         test_loss = test_loss/labels.shape[0]
#         print('Client ', client_id)
#         print(f'Test Loss: {test_loss:.6f}')
#         print(
#             f'Test Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
#             f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})\n'
#         )
#     pdb.set_trace()
#     return total_correct / total_vol

train()
# Final test
print('\n ### Final Test ###')
# test_part = MNISTPartitioner(
#     test_data.targets,
#     num_clients=num_clients,
#     partition="noniid-#label",
#     major_classes_num=major_classes_num,
#     seed = seed
# )
pdb.set_trace()

# Send the user model's gradient to the server
# user_gradient = {param_name: param.grad for param_name, param in user_model.named_parameters()}
# torch.distributed.rpc.rpc_sync("client_{}".format(client_id), server_model.update_gradient, args=(user_gradient,))

# # Aggregate the gradients and update the server model
# server_gradient = server_model.aggregate_gradients()
# server_optimizer.zero_grad()
# for param_name, param in server_model.front_part.named_parameters():
#     if param_name in server_gradient:
#         param.grad = server_gradient[param_name]
# server_optimizer.step()

    