import torch
import torch.nn as nn
import torch.optim as optim

# 定义两个神经网络部分 f1 和 f2
class F1(nn.Module):
    def __init__(self):
        super(F1, self).__init__()
        self.layer = nn.Linear(10, 20)  # 示例层

    def forward(self, x):
        return self.layer(x)

class F2(nn.Module):
    def __init__(self):
        super(F2, self).__init__()
        self.layer = nn.Linear(20, 1)  # 示例层

    def forward(self, x):
        return self.layer(x)

# 创建实例
f1 = F1()
f2 = F2()

# 定义优化器
optimizer_f2 = optim.SGD(f2.parameters(), lr=0.01)
optimizer_f1 = optim.SGD(f1.parameters(), lr=0.01)

# 定义损失函数
criterion = nn.MSELoss()

# 假设我们有一个 batch 的输入 x 和对应的标签 y_true
x = torch.randn(32, 10)  # batch size = 32, input size = 10
y_true = torch.randn(32, 1)  # target size = 1

# 前向传播 f1
z = f1(x)
y_pred = f2(z)

# 用于存储对 z 的梯度
grads_z = torch.zeros_like(z)

# 计算每个样本的损失和梯度
losses = []
for i in range(x.size(0)):
    z_i = z[i:i+1].detach().clone().requires_grad_(True)
    y_pred_i = f2(z_i)
    loss_i = criterion(y_pred_i, y_true[i:i+1])
    loss_i.backward()
    grads_z[i:i+1] = z_i.grad
    losses.append(loss_i.item())

# 计算所有样本的损失平均值
avg_loss = sum(losses) / len(losses)

# 反向传播更新 f2
optimizer_f2.zero_grad()
avg_loss_tensor = torch.tensor(avg_loss, dtype=torch.float32, requires_grad=True)
avg_loss_tensor.backward()
optimizer_f2.step()

# 对每个样本的梯度进行剪裁
clipped_grads_z = torch.clamp(grads_z, min=-1.0, max=1.0)

# 计算剪裁后梯度的均值并加入高斯噪声
mean_grad_z = torch.mean(clipped_grads_z, dim=0)
stddev = 0.1
noisy_mean_grad_z = mean_grad_z + stddev * torch.randn_like(mean_grad_z)

# 将噪声后的均值梯度扩展到整个 batch
noisy_mean_grad_z_expanded = noisy_mean_grad_z.unsqueeze(0).expand_as(z)

# 反向传播更新 f1
optimizer_f1.zero_grad()
z.backward(noisy_mean_grad_z_expanded)

# 更新 f1
optimizer_f1.step()
