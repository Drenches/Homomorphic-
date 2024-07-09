import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pdb

# 定义LeNet-5的卷积部分 f1
class F1(nn.Module):
    def __init__(self):
        super(F1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten the tensor
        return x

# 定义LeNet-5的全连接部分 f2
class F2(nn.Module):
    def __init__(self):
        super(F2, self).__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def prune_tensor_to_fixed_norm(tensor, fixed_norm):
    # 计算原张量的二范数
    original_norm = np.linalg.norm(tensor)
    
    # 如果原张量的二范数小于等于固定值，直接返回原张量
    if original_norm <= fixed_norm:
        return tensor
    
    # 计算缩放因子
    scale_factor = fixed_norm / original_norm
    
    # 缩放张量
    pruned_tensor = tensor * scale_factor
    
    return pruned_tensor

# 创建实例
f1 = F1()
f2 = F2()

# 定义优化器
optimizer_f2 = optim.SGD(f2.parameters(), lr=0.01)
optimizer_f1 = optim.SGD(f1.parameters(), lr=0.01)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 加载MNIST数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
delta = 1e-5
sigma = 1


# # 训练过程
for epoch in range(3):  # 仅训练一个epoch作为示例
    for x, y_true in trainloader:
        # 前向传播 f1
        # z = f1(x)
        # y_pred = f2(z)

        # 用于存储对 z 的梯度
        # grads_zs = torch.zeros_like(z)
        
        batch_loss = 0.0
        optimizer_f2.zero_grad()
        optimizer_f1.zero_grad()
        for i in range(x.size(0)):
            z_i = f1(x[i:i+1])
            y_pred_i = f2(z_i)
            loss_i = criterion(y_pred_i, y_true[i:i+1])
            batch_loss += loss_i

        # pdb.set_trace()
        # 计算所有样本的损失平均值
        batch_loss /= len(x)

        # 反向传播更新 f2
        # optimizer_f2.zero_grad()
        batch_loss.backward()
        optimizer_f2.step()
        
        # 更新 f1
        optimizer_f1.step()

        print(f'Epoch [{epoch+1}], Loss: {batch_loss.detach().item()}')

print('Finished Training')


# 训练过程
# for epoch in range(3):  # 仅训练一个epoch作为示例
#     for x, y_true in trainloader:
#         z = f1(x)
#         y_pred = f2(z)
#         loss= criterion(y_pred, y_true)
#         optimizer_f2.zero_grad()
#         optimizer_f1.zero_grad()
#         loss.backward()
#         optimizer_f2.step()
#         optimizer_f1.step()
#         print(f'Epoch [{epoch+1}], Loss: {loss.detach().item()}')

# print('Finished Training')



# loss_i.backward()
# grads_z = z_i.grad
# grads_z  = prune_tensor_to_fixed_norm(grads_z, 4)# 对每个样本的梯度进行剪裁
# grads_zs[i:i+1] = grads_z 


# 计算剪裁后梯度的均值并加入高斯噪声
# mean_grad_z = torch.mean(grads_zs, dim=0)
# stddev = sigma*delta
# noisy_mean_grad_z = mean_grad_z 
# pdb.set_trace()
# + stddev * torch.randn_like(mean_grad_z)
# 将噪声后的均值梯度扩展到整个 batch
# noisy_mean_grad_z_expanded = noisy_mean_grad_z.unsqueeze(0).expand_as(z)

# 反向传播更新 f1

# z.backward(noisy_mean_grad_z_expanded)
# z.backward(grads_zs)
# z.backward()