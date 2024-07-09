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


## Test mode setting
iid = False
Pure_FedAvg = False

## Load data and model
gpu = torch.cuda.is_available()
total_num_classes = 10
batch_size = 64
num_clients = 100
cut_point = 18
adaptiveLR = False
# seed = 2021
data_name = 'CIFAR10'
if iid:
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, DATA_CLASS = load_partition_data_cifar10('data', num_clients, 10, batch_size)
else:
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, DATA_CLASS = load_partition_data_cifar10('data', num_clients, 2, batch_size)

# User model list
# client_model_list = [CIFAR10CNNUser().cuda().train() for _ in range(num_clients)]
client_model_list = [SimpleUserNet().cuda().train() for _ in range(num_clients)]
# client_model_list = [UserNetCIFAR10().cuda().train() for _ in range(num_clients)]
# client_model_list = [DynamicSplitCIFAR10CNN(cut_point=cut_point, server=False).cuda().train() for _ in range(num_clients)]

# Server model initi
# globe_server_model = CIFAR10CNNServer().cuda()
globe_server_model = SimpleServerNet().cuda()
# globe_server_model = ServerNetCIFAR10().cuda().train()
# globe_server_model = DynamicSplitCIFAR10CNN(cut_point=cut_point, server=True).cuda()

def adaptiverl(round, adaptive=True): 
    """
    return: clientLearingRate, serverLearingRate
    """
    if adaptive:
        if round<200:
            return 1e-4, 1e-2
        else:
            return 1e-2, 1e-5
    else:
        return 1e-3, 1e-3

# Muti-round training
def train(round = 200, p=0.1):

    learningRate = 1e-3
    eps = 1e-3
    AMSGrad = True
    server_model_list = [copy.deepcopy(globe_server_model) for _ in range(num_clients)]
    # server_optimizer_list = [torch.optim.Adam(params = server_model.parameters(), lr = learningRate) for server_model in server_model_list]
    # client_optimizer_list = [torch.optim.Adam(params = client_model.parameters(), lr = learningRate) for client_model in client_model_list]
    # server_optimizer_list = [torch.optim.Adam(params = server_model.parameters(), eps=eps, amsgrad=AMSGrad, lr = learningRate) for server_model in server_model_list]
    # client_optimizer_list = [torch.optim.Adam(params = client_model.parameters(), eps=eps, amsgrad=AMSGrad, lr = learningRate) for client_model in client_model_list]
    client_list = [i for i in range(num_clients)]
    criterion = nn.CrossEntropyLoss()
    

    for i in range(round):
        print('\nround:', i)
        # randomly select a subset of clients
        random_selected_clients_list = random.sample(client_list, int(len(client_list)*p))
        avg_train_acc = []
        clientLearingRate, serverLearingRate = adaptiverl(round, adaptive=adaptiveLR)
        for client_id in random_selected_clients_list:
            # client_id = 0
            # Train multiple epoches for selected clients
            # Load client's dataset
            client_dataloader = train_data_local_dict[client_id]
            # client_optimizer = client_optimizer_list[client_id]
            client_model = client_model_list[client_id].train()

            # Load server model corresonding for each specific client
            server_model = server_model_list[client_id].train()
            # server_optimizer =server_optimizer_list[client_id]
            
            client_optimizer = torch.optim.Adam(params = client_model.parameters(), eps=eps, amsgrad=AMSGrad, lr = clientLearingRate)
            server_optimizer = torch.optim.Adam(params = server_model.parameters(), eps=eps, amsgrad=AMSGrad, lr = serverLearingRate)

            correct = 0
            total = 0
            train_loss = 0
            # epoches = random.randint(2,5)
            epoches = 5
            for epoch in range(epoches):

                # Train the user model on the client's dataset
                for _, (images, labels) in enumerate(client_dataloader):

                    client_optimizer.zero_grad()
                    server_optimizer.zero_grad()

                    images, labels = images.permute(0, 3, 1, 2).cuda(), labels.cuda()

                    # Forward pass through the server model
                    front_output = server_model(images)
                    
                    # Forward pass through the user model
                    user_output = client_model(front_output)

                    # Calculate the loss and perform backpropagation
                    loss = criterion(user_output, labels)
                    loss.backward()

                    client_optimizer.step()
                    server_optimizer.step()

                    _, pred = torch.max(user_output, 1)
                    correct += torch.sum(np.squeeze(pred.eq(labels.data.view_as(pred))))
                    train_loss += loss.item()
                    total += labels.shape[0]

            # Calculate the train accuracy of each client
            # print('Client ', client_id)
            # print('Train Loss: %.4f,  Train Acc.: %.4f' % (train_loss/total , correct/total))
            avg_train_acc.append(correct/total)
        
        # print('Avg. Train Acc. over Clients: %4f' % ( sum(avg_train_acc).item()/len(random_selected_clients_list) ))
        
        # pdb.set_trace()
        # Aggregate and update the server model
        updated_server_models = [server_model_list[i] for i in random_selected_clients_list]
        new_global_model = aggregation(updated_server_models)
        for j in range(num_clients):
            server_model_list[j].load_state_dict(new_global_model.state_dict()) 
        
        # if i%2==0 and i!=0:
        # updated_client_models = [client_model_list[i] for i in random_selected_clients_list]
        # new_global_client_model = aggregation(updated_client_models)
        # for j in range(num_clients):
        #     client_model_list[j].load_state_dict(new_global_client_model.state_dict())

        # test(distribution_list=distribution_list.copy(), server_model_list= server_model_list)
        if i%2==0 and i!=0:
            with torch.no_grad():
                test_acc = []
                for client_id in random_selected_clients_list:
                # for client_id in range(num_clients):
                    # client_id = 0
                    test_loader = test_data_local_dict[client_id]
                    correct = 0
                    total = 0
                    test_loss = 0
                    for images, labels in test_loader:
                        images, labels = images.permute(0, 3, 1, 2).cuda(), labels.cuda()
                        server_model = server_model_list[client_id].eval()
                        client_model = client_model_list[client_id].eval()
                        front_ouput = server_model(images)
                        output = client_model(front_ouput)
                        loss = criterion(output, labels)
                        _, predicted = torch.max(output.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        test_loss += loss.item()
                    # print('Client ', client_id)
                    # print('Test Loss: %.4f,  Test Acc.: %.4f\n' % (train_loss/total , correct/total))
                    test_acc.append(correct/total)

                print('\n### Avg. Test Acc. over Clients: %4f ###' % ( sum(test_acc)/len(random_selected_clients_list) ))
    
    return new_global_model, client_model_list

def tuning(model1, model2, train_loader, test_loader, num_epoches = 20):
    model1.train()
    model2.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
    for _ in range(num_epoches):
        for batch_idx, (data, target) in enumerate(train_loader):
            # if dataset == 'cifar10':
            data, target = data.permute(0, 3, 1, 2).cuda(), target.cuda()
            # else:
                # data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            x = model1(data)
            output = model2(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
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

def save_weights(model, path_id, root_path='/home/dev/workspace/Homomorphic-HalfFed/saved_weights/50/'):
    path = root_path + str(path_id)+'.pth'
    torch.save(model.state_dict(), path)


global_model, client_models = train()
pdb.set_trace()
save_weights(global_model, 0)
for i in [i for i in range(num_clients)]:
    save_weights(client_models[i], i+1)

# pdb.set_trace()
client_list = [i for i in range(num_clients)]
selected_client_id  = random.sample(client_list, 1)[0]
model_for_tuning = client_models[selected_client_id]
train_loader =  train_data_local_dict[selected_client_id]
test_loader = test_data_local_dict[selected_client_id]
tuning(global_model, model_for_tuning, train_loader, test_loader)

        # print('server model aggregated')
        
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

# if data_name== 'CIFAR10':
#     train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
#     test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
#     # Split the dataset into multiple clients
    
#     #iid
#     # client_datasets_part  = CIFAR10Partitioner(train_data.targets,
#     #                                   num_clients,
#     #                                   balance=True,
#     #                                   partition="iid",
#     #                                   seed=seed)

#     # shard division
#     num_shards = 100
#     client_datasets_part = CIFAR10Partitioner(
#         train_data.targets,
#         num_clients=num_clients,
#         balance=None,
#         partition="shards",
#         num_shards=num_shards,
#         seed = seed
#     )

#     # dir alpha class division
#     # client_datasets_part = CIFAR10Partitioner(train_data.targets, 
#     #                                     num_clients=num_clients,
#     #                                     balance=None,
#     #                                     partition="dirichlet", 
#     #                                     dir_alpha=0.5,
#     #                                     seed=seed)
    
#     # load server model
#     globe_server_model = ServerNetCIFAR10().cuda()
#     # server_model_list = [copy.deepcopy(server_model_init) for _ in range(num_clients)]
#     # server_model_list = [ServerNetCIFAR10().cuda().train() for _ in range(num_clients)]
    