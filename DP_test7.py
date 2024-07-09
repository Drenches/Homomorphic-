import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from opacus.accountants import RDPAccountant
import logging
from datetime import datetime
import os
import pdb

# 创建保存日志的文件夹

class F1(nn.Module):
    def __init__(self):
        super(F1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return x

class F2(nn.Module):
    def __init__(self):
        super(F2, self).__init__()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def init_log(log, log_dir):
    if log:
        os.makedirs(log_dir, exist_ok=True)
        current_file_name = os.path.splitext(os.path.basename(__file__))[0]
        log_file = os.path.join(log_dir, f'{current_file_name}_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        print(f"Log file path: {log_file}")  # Debugging line
        return log_file


def calculate_epsilon(q, sigma, delta, steps):
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=sigma, sample_rate=q)
        epsilon, _ = accountant.get_privacy_spent(delta=delta)
        print(epsilon)
    return epsilon

def gaussian_mechanism(tensor, max_norm, sigma, delta):
    with torch.no_grad():
        noise = torch.randn(tensor.size()) * sigma * max_norm
        noise = noise.cuda()
    return tensor + noise


def per_sample_norm_clip(tensor, fixed_norm):
    """
    Prunes the tensor to have a fixed norm along a specified dimension.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        fixed_norm (float): The desired fixed norm.
        batch_dim (int): The batch dimension index. Default is 0.

    Returns:
        torch.Tensor: The pruned tensor with a fixed norm.
    """
    original_norms = torch.norm(tensor.view(tensor.size(0), -1), dim=1)
    scale_factors = fixed_norm / (original_norms + 1e-9)
    clipped_scale_factors = torch.minimum(scale_factors, torch.ones_like(scale_factors))
    pruned_tensor = tensor * clipped_scale_factors.unsqueeze(1)
    pruned_norms = torch.norm(pruned_tensor.view(pruned_tensor.size(0), -1), dim=1)
    return pruned_tensor

def per_sample_automatic_clip(tensor):
    """
    Advanced clip, cite from https://mdnice.com/writing/e36880200bf8400e8db8f0c35f4db356

    It's amazing and fanstic! It is highly recommanded for DP guys.
    """
    original_norms = torch.norm(tensor.view(tensor.size(0), -1), dim=1)
    scale_factors =  1 / (original_norms + 1e-2)
    pruned_tensor = tensor * scale_factors.unsqueeze(1)
    return pruned_tensor

def per_sample_adaptive_clip(tensor):
    """
    adaptive clip, modified automatic_clip

    However, in our application, this method is not good. It's very unstable
    """
    original_norms = torch.norm(tensor.view(tensor.size(0), -1), dim=1)
    scale_factors = 1/ ((1e-2 / original_norms + 1e-2) + original_norms)
    pruned_tensor = tensor * scale_factors.unsqueeze(1)
    return pruned_tensor

def total_norm_clip(tensor, fixed_norm):
    """
    Prunes the entire tensor to have a fixed norm.

    Args:
        tensor (torch.Tensor): The input tensor.
        fixed_norm (float): The desired fixed norm.

    Returns:
        torch.Tensor: The pruned tensor with a fixed norm.
    """
    original_norm = torch.norm(tensor)
    
    scale_factor = fixed_norm / (original_norm + 1e-9)
    
    if scale_factor > 1:
        return tensor
    else:
        # 缩放张量
        pruned_tensor = tensor * scale_factor
        return pruned_tensor


f1 = F1()
f2 = F2()

# 经过大量的测试，在MNIST这个数据上，得出基本最优的配置是：
# 使用automatic clipping
# 参数设置为：
# delta = 1e-5
# sigma = 0.5
# max_norm = 1
# batch_size = 16
# test_batch_size = 512
# learning_rate_f1 = 1e-2
# learning_rate_f2 = 1e-4
# 在这组参数下，10以内的隐私预算即可快速收敛到越97%的精度
# 注意，这里的max_norm是固定为1的，与手动裁剪不同，这里只需要调整f1的学习率这一个参数
delta = 1e-5
sigma = 0.5
max_norm = 1
batch_size = 16
test_batch_size = 512
learning_rate_f1 = 1e-2
learning_rate_f2 = 1e-4
data_name = 'mnist'
log = True
log_dir = '/home/dev/workspace/Homomorphic-HalfFed/logs'
log_file = init_log(log, log_dir)

# record 
init_log_message = f"traing batch size {batch_size}\
    \nsigma: {sigma:.4f}\nlearning rate (f1): {learning_rate_f1}, learning rate (f1): {learning_rate_f2}\n\
    max_norm: {max_norm}"
print(init_log_message)
if log:
    with open(log_file, 'a') as f:
        f.write(init_log_message + '\n')
accountant = RDPAccountant()

optimizer_f2 = optim.AdamW(f2.parameters(), lr=learning_rate_f1)
optimizer_f1 = optim.AdamW(f1.parameters(), lr=learning_rate_f2)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True)
sample_rate = batch_size/len(trainset)

step = 0
for epoch in range(1000):
    f1.train()
    f2.train()
    f1.cuda()
    f2.cuda()
    batch_round = 0
    for x, y_true in trainloader:
        optimizer_f1.zero_grad()

        x, y_true = x.cuda(), y_true.cuda()
        
        z = f1(x)
        z.retain_grad()
        y_pred = f2(z)
        loss = criterion(y_pred, y_true)
        loss.backward(retain_graph=True)
        optimizer_f2.zero_grad()

        # vanilla test
        optimizer_f1.zero_grad() #这里一定要有这一句，否则前面loss.backward的梯度没有清空
        batch_grad_z = z.grad.clone()
        # clipped_batch_grad_z = per_sample_norm_clip(batch_grad_z, max_norm)
        # clipped_batch_grad_z = total_norm_clip(batch_grad_z, max_norm)
        clipped_batch_grad_z = per_sample_automatic_clip(batch_grad_z)
        # clipped_batch_grad_z = per_sample_adaptive_clip(batch_grad_z)
        noisy_avg_batch_grad_z = gaussian_mechanism(clipped_batch_grad_z, max_norm, sigma, delta)
        # z.backward()
        # z.backward(batch_grad_z)
        z.backward(noisy_avg_batch_grad_z)
        optimizer_f1.step()

        batch_round += 1
        correct = (torch.argmax(y_pred, dim=1) == y_true).sum().item()

        accountant.step(noise_multiplier=sigma*max_norm, sample_rate=sample_rate)
        step += 1
        
    # print(f'Batch round {batch_round*batch_size}/60000')
    # print(f'Train acc {correct/len(trainset)}')
    # print(f'Epoch [{epoch+1}], Loss: {loss.detach().item()}')
    epsilon, _ = accountant.get_privacy_spent(delta=delta)
    print(f"After {step+1} steps, the privacy budget is (ε = {epsilon:.2f}, δ = {delta})")
        
    f1.cuda()
    f2.cuda()
    f1.eval()
    f2.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y_true in testloader:
            x, y_true = x.cuda(), y_true.cuda()
            z = f1(x)
            y_pred = f2(z)
            loss = criterion(y_pred, y_true)
            test_loss += loss.item() * x.size(0)
            correct += (torch.argmax(y_pred, dim=1) == y_true).sum().item()
            total += y_true.size(0)
    
    test_loss /= len(testloader.dataset)
    test_acc = 100 * correct / total

    # 打印和保存日志
    log_message = f"Epoch {epoch}\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n"
    print(log_message)
    if log:
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')


print('Finished Training')
