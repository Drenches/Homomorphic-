import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import tenseal as ts
from utils import *


class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        
    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SimpleConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(SimpleConvNet, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = self.flatten(x)
        # x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x

class Encrypt(torch.nn.Module):
    def __init__(self, torch_nn):
        super(Encrypt, self).__init__()        
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
  
    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        return enc_x
  
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ServerNetMNIST(torch.nn.Module):
    def __init__(self, hidden=64):
        super(ServerNetMNIST, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
  
    def forward(self, x):
        x = self.conv1(x)
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        return x
  
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class UserNetMNIST(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(UserNetMNIST, self).__init__()        

        self.fc2 = torch.nn.Linear(hidden, output)
  
    def forward(self, x):
        x = torch.relu(x)
        x = self.fc2(x)
        return x
  
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class LeNetServer(nn.Module):
    def __init__(self):
        super(LeNetServer, self).__init__()
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

class LeNetUser(nn.Module):
    def __init__(self):
        super(LeNetUser, self).__init__()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20Server(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20Server, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out


class ResNet20User(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20User, self).__init__()
        self.in_planes = 16

        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = nn.AvgPool2d(kernel_size=8)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

class ServerNetCIFAR10(nn.Module):
    def __init__(self):
        super(ServerNetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) #1
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(
            ResNetBlock(16, 16, stride=1), #3
            ResNetBlock(16, 16, stride=1) #5
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(16, 32, stride=2), #7
            ResNetBlock(32, 32, stride=1) #9
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(32, 64, stride=2), #11
            ResNetBlock(64, 64, stride=1) #13
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        return out


class UserNetCIFAR10(nn.Module):
    def __init__(self, output=10):
        super(UserNetCIFAR10, self).__init__()
        self.layer4 = nn.Sequential(
            ResNetBlock(16, 16, stride=1), #15
            # ResNetBlock(16, 16, stride=1), #17
            # ResNetBlock(16, 16, stride=1) #19
        )
        self.layer5 = nn.Sequential(
            ResNetBlock(16, 32, stride=2), #21
            # ResNetBlock(32, 32, stride=1), #23
            # ResNetBlock(32, 32, stride=1) #25
        )
        self.layer6 = nn.Sequential(
            ResNetBlock(32, 64, stride=2), #27
            # ResNetBlock(64, 64, stride=1), #29
            # ResNetBlock(64, 64, stride=1) #31
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output) 

    def forward(self, x):
        out = self.layer4(x)
        # out = self.layer5(x) #临时测试-on
        out = self.layer5(out)
        out = self.layer6(out) # 临时测试-off
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CombinedNet(nn.Module):
    def __init__(self, output=10):
        super(CombinedNet, self).__init__()
        self.server_net = ServerNetCIFAR10()
        self.user_net = UserNetCIFAR10(output=output)

    def forward(self, x):
        # 首先通过ServerNetCIFAR10的前向传播
        server_out = self.server_net(x)
        
        # 然后将ServerNet的结果作为UserNet的输入
        user_out = self.user_net(server_out)
        
        return user_out


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        print("NN: MLP is created")
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class SimpleServerNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=10):
        super(SimpleServerNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 6, 5)#定义第一个卷积层
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)#定义第二个卷积层
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#第一个卷积层激活并池化
        x = self.pool(F.relu(self.conv2(x)))#第二个卷积层激活并池化
        x = x.reshape(-1, 16 * 5 * 5)
        return x

class SimpleUserNet(nn.Module):
    def __init__(self,input_dim=3, output_dim=10):
        super(SimpleUserNet, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#定义第一个全连接
        self.fc2 = nn.Linear(120, 84)#定义第二个全连接
        self.fc3 = nn.Linear(84, 10)#定义第三个全连接

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class CombinedNet2(nn.Module):
    def __init__(self):
        super(CombinedNet2, self).__init__()
        self.server_net = SimpleServerNet()
        self.user_net = SimpleUserNet()

    def forward(self, x):
        # 首先通过ServerNetCIFAR10的前向传播
        server_out = self.server_net(x)
        
        # 然后将ServerNet的结果作为UserNet的输入
        user_out = self.user_net(server_out)
        
        return user_out

class CIFAR10CNNServer(nn.Module):
    def __init__(self, NChannels=3, NUM_CLASS=10):
        super(CIFAR10CNNServer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = NChannels, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.maxpool3 = nn.MaxPool2d(2,2)
        self.feature_dims = 4 * 4 * 128


    def forward(self, img):
        x = nn.ReLU()(self.conv1(img))
        x = nn.ReLU()(self.conv2(x))
        x = self.maxpool1(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = self.maxpool2(x)
        x = nn.ReLU()(self.conv5(x))
        x = nn.ReLU()(self.conv6(x))
        x = self.maxpool3(x)
        x = x.reshape(-1, self.feature_dims)

        return x

class CIFAR10CNNUser(nn.Module):
    def __init__(self, NChannels=3, NUM_CLASS=10):
        super(CIFAR10CNNUser, self).__init__()
        self.feature_dims = 4 * 4 * 128
        self.fc = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.feature_dims, 512)),
            ('sig1',nn.Sigmoid()),
            ('fc2',nn.Linear(512, NUM_CLASS))
        ]))

    def forward(self, x):

        # x = x.view(-1, self.feature_dims)
        
        x = self.fc(x)

        return x

class CIFAR10CNN(nn.Module):
    def __init__(self, NChannels=3, NUM_CLASS=10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = NChannels, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.maxpool3 = nn.MaxPool2d(2,2)
        
        
        self.feature_dims = 4 * 4 * 128
        self.fc = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.feature_dims, 512)),
            ('sig1',nn.Sigmoid()),
            ('fc2',nn.Linear(512, NUM_CLASS))
        ]))

    def forward(self, img):
        x = nn.ReLU()(self.conv1(img))
        x = nn.ReLU()(self.conv2(x))
        x = self.maxpool1(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = self.maxpool2(x)
        x = nn.ReLU()(self.conv5(x))
        x = nn.ReLU()(self.conv6(x))
        x = self.maxpool3(x)

        x = x.reshape(-1, self.feature_dims)
        x = self.fc(x)

        return x

class DynamicSplitCIFAR10CNN(nn.Module):
    def __init__(self,  cut_point, server = True,NChannels=3, NUM_CLASS=10):
        super(DynamicSplitCIFAR10CNN, self).__init__()
        
        self.server = server
        self.cut_point = cut_point
        self.feature_dims = 4 * 4 * 128  # Assuming this is the output dimension after the last convolution
        
        # Define the entire network architecture
        self.all_layers = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(in_channels=NChannels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # Flatten for fully connected layers
            nn.Flatten(),
            nn.Linear(self.feature_dims, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASS)
        )
        
        # Splitting the network based on cut_point
        self.edge_partition, self.server_partition = torch.nn.Sequential(), torch.nn.Sequential()
        for i, layer in enumerate(self.all_layers):
            if i < self.cut_point:
                self.server_partition.add_module(f'layer_{i}', layer)
            else:
                self.edge_partition.add_module(f'layer_{i}', layer)
    
    def forward(self, x):
        if self.server:

            x = self.server_partition(x)
        else:

            x = self.edge_partition(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        print("NN: MLP is created")
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#定义第一个卷积层
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)#定义第二个卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#定义第一个全连接
        self.fc2 = nn.Linear(120, 84)#定义第二个全连接
        self.fc3 = nn.Linear(84, 10)#定义第三个全连接

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#第一个卷积层激活并池化
        x = self.pool(F.relu(self.conv2(x)))#第二个卷积层激活并池化
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        # x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, unsqueeze=True, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.unsqueeze = unsqueeze
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.unsqueeze:
            x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x


def load_model(args):
    if args.dataset_name == "mnist":
         client_model_list = [UserNetMNIST().train() for _ in range(args.client_num_in_total)]
         server_model = ServerNetMNIST().train()
    elif args.dataset_name == "fashion-mnist":
         client_model_list = [LeNetUser().train() for _ in range(args.client_num_in_total)]
         server_model = LeNetServer().train()
    elif args.dataset_name == "cifar10":
        client_model_list = [CIFAR10CNNUser().train() for _ in range(args.client_num_in_total)]
        server_model = CIFAR10CNNServer().train()
    else:
        raise ValueError(f"dataset {args.dataset_name} have not been supported.")
    
    return client_model_list, server_model



# class ResNetBlock(nn.Module):
# # this implementation of ResBlock considers split a resblock into two parties(server and client)
#     def __init__(self, in_channels, out_channels, stride=1, server_part=False):
#         super(ResNetBlock, self).__init__()
#         self.server_part = server_part
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         if self.server_part:
#             out = self.conv1(x)
#             out = self.bn1(out)
#             out = F.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#         else:
#             out = F.relu(x)
#             out = self.conv1(out)
#             out = self.bn1(out)
#             out = F.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#         shortcut = self.shortcut(x)
#         out += shortcut
#         out = F.relu(out)
#         return out