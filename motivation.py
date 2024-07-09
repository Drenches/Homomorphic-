from nets import *
from ptflops import get_model_complexity_info
import pdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams


def convert_to_flops(flops_str):
    units = {'GMac': 1e9, 'MMac': 1e6, 'KMac': 1e3}
    value, unit = flops_str.split()
    return float(value) * units[unit]

def convert_to_params(params_str):
    value, unit = params_str.split()
    if unit == 'M':
        return float(value) * 1e6
    elif unit == 'K':
        return float(value) * 1e3
    else:
        return float(value)

num_clients = 1000
sample_rate = 1
num_samples = 60000/num_clients
num_epoches = 2
batch_size = 16
num_round = num_samples/batch_size
total_round = 10
# algo = "FedAvg"
# algo = "FedPer"
algo = "TailorFL"


mnist_net = SimpleConvNet()
emnist_net = LeNet5()
cifar10_net = CombinedNet2()


datasets = ['MNIST', 'Fashion-MNIST', 'CIFAR10']
models = [mnist_net, emnist_net, cifar10_net]
modes = ['FedAvg', 'FedSGD']
mode = modes[0]
local_mul_ops = []
local_add_ops = []
agg_add_ops = []
colors = plt.get_cmap('Set2').colors

for i, model in enumerate(models):
    if i == 2:
        input_shape = (3, 32, 32)
    elif i == 0:
        input_shape = (1, 28, 28)
    else:
        input_shape = (28, 28)
    # input = torch.randn(1, dim, 32, 32)
    flops_str, params_str = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {flops_str}")
    print(f"Params: {params_str}")
    flops = convert_to_flops(flops_str)
    params = convert_to_params(params_str)
    total_mul_ops = flops
    total_add_ops = flops 
    if algo == 'FedAvg':
        local_flops = total_mul_ops * 3 * num_samples * num_epoches * total_round
        server_flops = params * num_clients * sample_rate * total_round
        ratio_server = server_flops/(local_flops+server_flops)
        local_add_ops.append(total_mul_ops * 3 * num_samples * num_epoches * total_round)
        local_mul_ops.append(total_mul_ops * 3 * num_samples * num_epoches * total_round)
        agg_add_ops.append(params * num_clients * sample_rate * total_round)
    elif algo == 'FedSGD':
        local_add_ops.append(total_mul_ops * 3 * batch_size * total_round)
        local_mul_ops.append(total_mul_ops * 3 * batch_size * total_round)
        agg_add_ops.append(params * num_clients * sample_rate * total_round)
    elif algo == 'FedPer':
        local_flops = total_mul_ops * 3 * num_samples * num_epoches * total_round
        server_flops = params * num_clients * sample_rate * total_round * 3/5
        ratio_server = server_flops/(local_flops+server_flops)
        local_add_ops.append(total_mul_ops * 3 * batch_size * total_round)
        local_mul_ops.append(total_mul_ops * 3 * batch_size * total_round)
        agg_add_ops.append(params * num_clients * sample_rate * total_round * 2/3)
    elif algo == 'TailorFL':
        local_flops = total_mul_ops * 3 * num_samples * num_epoches * total_round * 1/3
        server_flops = params * num_clients * sample_rate * total_round * 3
        ratio_server = server_flops/(local_flops+server_flops)
        local_add_ops.append(total_mul_ops * 3 * batch_size * total_round)
        local_mul_ops.append(total_mul_ops * 3 * batch_size * total_round)
        agg_add_ops.append(params * num_clients * sample_rate * total_round * 2/3)
    # 3: 1 for forward pg and 2 for backward pg during training
    print(f"{algo}")
    print(f"Local Flops: {local_flops}")
    print(f"Server Flops: {server_flops}")
    print(f"Server ratio: {ratio_server}\n")
    print(f"Local ratio: {1-ratio_server}\n")
    # print(f"{(total_mul_ops * 3 * num_samples * num_epoches*2) /(params * num_clients * sample_rate)} x")

pdb.set_trace()
# Data preparation
N = len(datasets)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

# Plotting the bar chart
fig, ax = plt.subplots()

# Set global font size
plt.rcParams.update({
    'font.size': 16,               # General font size
    'axes.titlesize': 16,          # Axes title font size
    'axes.labelsize': 16,          # Axes labels font size
    'xtick.labelsize': 16,         # X-axis tick labels font size
    'ytick.labelsize': 16,         # Y-axis tick labels font size
    'legend.fontsize': 12,         # Legend font size
    # 'figure.titlesize': 16         # Figure title font size
})

# Local computation
p1 = ax.bar(ind - width/2, local_mul_ops, width, label='Local workload (FlOPs)', edgecolor=colors[0], hatch='...', fill=False)
# p2 = ax.bar(ind - width/2, local_add_ops, width, bottom=local_mul_ops, label='Local comp. - Add', edgecolor='red', hatch='//', fill=False)

# Aggregation computation
p3 = ax.bar(ind + width/2, agg_add_ops, width, label='Server workload (Addition)', edgecolor=colors[1], hatch='//', fill=False)

ax.axhline(0, color='grey', linewidth=0.8)
ax.set_ylabel('Number of operations', fontsize = 16)
# ax.set_title('Comparison of operations between local training and aggregation')
ax.set_xticks(ind)
ax.set_yscale('log')
# ax.grid(True)
ax.set_xticklabels(datasets, fontsize = 16)
ax.legend(loc='upper left')

# # Adding data labels
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(int(height)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(p1)
# autolabel(p2)
# autolabel(p3)

root_save_path = '/home/dev/workspace/Homomorphic-HalfFed/figs/'
plt.savefig(root_save_path + "motivation_"+ algo + ".jpeg")
pdf = PdfPages(root_save_path + "motivation_"+ algo  + '.pdf')
pdf.savefig(bbox_inches = 'tight', pad_inches = 0)
pdf.close()
plt.close()

