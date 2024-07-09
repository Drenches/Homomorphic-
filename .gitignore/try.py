import torch
from torchvision import datasets
import torchvision.transforms as transforms
from utils import *
import pdb
from torch.utils.data import WeightedRandomSampler

w = torch.tensor([69.0, 132.0, 55.0, 0.0, 74.0, 210.0, 113.0, 124.0, 247.0, 0.0])
num_class = 10

test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

class_counts = torch.bincount(test_data.targets)
class_weights = 1.0 / class_counts
re_w = class_weights*w
pdb.set_trace()
re_re_w = re_w[test_data.targets]

sampler = WeightedRandomSampler(weights=re_re_w, replacement=True, num_samples=len(test_data)//10)
test_loader = torch.utils.data.DataLoader(test_data, sampler=sampler, batch_size=48)

for images, labels in test_loader:
    images, labels = images.cuda(), labels.cuda()
    pdb.set_trace()


## Thanks to Chatgpt. It helps me understand how to use WeightedRandomSampler. This is a good example.