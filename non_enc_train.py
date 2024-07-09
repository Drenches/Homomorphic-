import torch
import numpy as np
from utils import *
from nets import load_model
from cifar10_data_loader import *
import pdb
from opacus.accountants import RDPAccountant
import datetime
import random
import argparse
import logging

"""
A quick simulation to verify whether the main idea of HyperColt is feasible. 
In this version, the data is not encrypted.
"""

parser = argparse.ArgumentParser(description="Non-encrypted training simulation")

parser.add_argument('--dataset_name', type=str, default='mnist', help='Name of the dataset to be used for training.')
parser.add_argument('--data_dir', type=str, default='/home/dev/workspace/data/', help='Name of the dataset to be used for training.')
parser.add_argument('--learning_rate_f1', type=float, default=1e-2, help='Learning rate for the server optimizer.')
parser.add_argument('--learning_rate_f2', type=float, default=1e-4, help='Learning rate for the client optimizer.')
parser.add_argument('--batch_size', type=int, default=8, help='Number of samples per batch.')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs for one aggregation.')
parser.add_argument('--total_rounds', type=int, default=100, help='Number of total runing round (one is a batch).')
parser.add_argument('--sigma', type=float, default=0.5, help='Standard devation for noise injector.')
parser.add_argument('--delta', type=float, default=1e-5, help='Relax factor for differential private.')
parser.add_argument('--client_num_in_total', type=int, default=100, help='Number of clients in setup. More than 10')
parser.add_argument('--client_data_class', type=int, default=5, help='Number of classes each client has. Less than 10')
parser.add_argument('--sample_rate', type=float, default=0.1, help='Sample rate for sample participations in each training round.')
parser.add_argument('--save_model', type=bool, default=False, help='Whether to save the trained model.')

args = parser.parse_args()

from data_preprocessing.process.DividedDataset import DividedDataset
dataset = DividedDataset(args)
dataset.load_data()
num_data_each = dataset.num_train//args.client_num_in_total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client_model_list, globe_server_model = load_model(args)


def train(args):

    server_model_list = [copy.deepcopy(globe_server_model) for _ in range(args.client_num_in_total)]
    client_list = [i for i in range(args.client_num_in_total)]
    criterion = nn.CrossEntropyLoss()
    
    for i in range(args.total_rounds):
        print('\nround:', i)

        random_selected_clients_list = random.sample(client_list, int(len(client_list)*args.sample_rate))
        for client_id in random_selected_clients_list:
            print(f'Client {client_id+1} is now training')

            client_dataloader = dataset.train_loader[client_id]
            client_model = client_model_list[client_id].train().to(device)
            server_model = server_model_list[client_id].train().to(device)
            
            client_optimizer = torch.optim.AdamW(params = client_model.parameters(), lr = args.learning_rate_f1)
            server_optimizer = torch.optim.AdamW(params = server_model.parameters(), lr = args.learning_rate_f2)
            
            step = 0
            correct = 0
            train_loss = 0

            for epoch in range(args.num_epochs):

                for images, labels in client_dataloader:

                    if args.dataset_name == 'cifar10':
                        images, labels = images.permute(0, 3, 1, 2).to(device), labels.to(device)
                        front_output = server_model(images)
                    else:
                        images, labels = images.to(device), labels.to(device)
                        front_output = server_model(images.unsqueeze(1))

                        
                    front_output.retain_grad()
                    user_output = client_model(front_output)
                    
                    client_optimizer.zero_grad()
                    loss = criterion(user_output, labels)
                    loss.backward(retain_graph=True)
                    train_loss += loss.detach().item()
                    client_optimizer.step()
                    

                    server_optimizer.zero_grad()
                    batch_grad_z = front_output.grad.clone()
                    clipped_batch_grad_z = per_sample_automatic_clip(batch_grad_z)
                    noisy_avg_batch_grad_z = gaussian_mechanism(clipped_batch_grad_z, 1, args.sigma, args.delta)
                    # front_output.backward(batch_grad_z)
                    front_output.backward(noisy_avg_batch_grad_z)
                    server_optimizer.step() # Here, maybe we can also try to accumulate gradient from multiple round and update 
                    
                    correct += (torch.argmax(user_output, dim=1) == labels).sum().item()
                    step += 1
                    
                    
                    # print(f'Batch round {step*args.batch_size}/{(dataset.num_train // args.client_num_in_total)}')
                    # print(f'Train acc {correct/args.batch_size}')
                    # print(f'Loss: {loss.detach().item()}')
                    # train_loss /= local_round
                    # train_acc = correct/(dataset.num_train * args.num_epochs)
                    # print(f'Train acc {train_acc}')
                    # print(f'Train loss {train_loss}')
            

            train_loss = train_loss / (num_data_each*args.num_epochs)
            train_acc = correct/(step*args.batch_size)
            print(f'Train acc {train_acc}')
            print(f'Train loss {train_loss}\n') 

            
        # Aggregate and update the server model
        updated_server_models = [server_model_list[i] for i in random_selected_clients_list]
        new_global_model = aggregation(updated_server_models)
        for j in range(args.client_num_in_total):
            server_model_list[j].load_state_dict(new_global_model.state_dict()) 
        

        # test(distribution_list=distribution_list.copy(), server_model_list= server_model_list)
        if i%2==0 and i!=0:
            with torch.no_grad():
                test_acc = []
                for client_id in random_selected_clients_list:
                    # client_id = 0
                    test_loader = dataset.test_loader[client_id]
                    correct = 0
                    total = 0
                    test_loss = 0
                    for images, labels in test_loader:
                        server_model = server_model_list[client_id].eval()
                        client_model = client_model_list[client_id].eval()
                        if args.dataset_name == 'cifar10':
                            images, labels = images.permute(0, 3, 1, 2).to(device), labels.to(device)
                            front_output = server_model(images)
                        else:
                            images, labels = images.to(device), labels.to(device)
                            front_output = server_model(images.unsqueeze(1))
                            
                        output = client_model(front_output)
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

global_model, client_models = train(args)


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



# pdb.set_trace()
# save_weights(global_model, 0)
# for i in [i for i in range(args.client_num_in_total)]:
#     save_weights(client_models[i], i+1)

# # pdb.set_trace()
# client_list = [i for i in range(args.client_num_in_total)]
# selected_client_id  = random.sample(client_list, 1)[0]
# model_for_tuning = client_models[selected_client_id]
# train_loader =  train_data_local_dict[selected_client_id]
# test_loader = test_data_local_dict[selected_client_id]
# tuning(global_model, model_for_tuning, train_loader, test_loader)

# pdb.set_trace()
