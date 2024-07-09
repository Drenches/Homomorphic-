import torch
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
import random
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torch.nn as nn
import torch.optim as optim
from nets import *

class GroupSampler(Sampler):
    def __init__(self, data_source, group_size):
        self.data_source = data_source
        self.group_size = group_size
        self.indices = list(range(len(self.data_source)))
    
    def __iter__(self):
        # random.shuffle(self.indices)
        grouped_indices = [self.indices[i:i + self.group_size] for i in range(0, len(self.indices), self.group_size)]
        random.shuffle(grouped_indices)
        flattened_indices = [idx for group in grouped_indices for idx in group]
        return iter(flattened_indices)
    
    def __len__(self):
        return len(self.data_source)

class GroupBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

def get_model(dataset_name):
    if dataset_name == 'MNIST':
        model = SimpleConvNet()
    elif dataset_name == 'CIFAR10' :
        model = CombinedNet()
    elif dataset_name == 'FashionMNIST':
        model = LeNet5()
    return model.cuda()


def get_datasets_and_input_size(dataset_name):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        input_size = 28 * 28
        num_classes = 10
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_size = 32 * 32 * 3
        num_classes = 10
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
        input_size = 28 * 28
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset")
    return train_dataset, test_dataset, input_size, num_classes


# 定义训练过程
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 定义测试过程
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累加batch的损失
            pred = output.argmax(dim=1, keepdim=True)  # 找到概率最大的类
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')


# 加载数据集
dataset_name = 'FashionMNIST'  # 修改为 'FashionMNIST', 'MNIST' 或 'CIFAR10'
train_dataset, test_dataset, input_size, num_classes = get_datasets_and_input_size(dataset_name)

# GroupSampler，设定每组大小为 n，例如 5
group_size = 1
sampler = GroupSampler(train_dataset, group_size)

# DataLoader
batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_sampler=GroupBatchSampler(sampler, batch_size, drop_last=False))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =  get_model(dataset_name)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 训练和测试模型
num_epochs = 50 
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)
