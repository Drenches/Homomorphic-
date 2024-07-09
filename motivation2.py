from nets import *
from ptflops import get_model_complexity_info
import pdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

# data source: @inproceedings{sphinx,
#   title={Sphinx: Enabling privacy-preserving online learning over the cloud},
#   author={Tian, Han and Zeng, Chaoliang and Ren, Zhenghang and Chai, Di and Zhang, Junxue and Chen, Kai and Yang, Qiang},
#   booktitle={Proc. IEEE S {\&} P},
#   pages={2487--2501},
#   year={2022},
# }
datasets = ['MNIST', 'CIFAR10']
batch_size = [500, 200]
training_runtimes = [390, 4276]
training_comm_per_image = [18.3, 20.0]
inference_runtimes = [6.01+500*0.05, 48.3+200*0.08]
inference_comm_per_image = [0.07, 1.46]

# Fig colors
colors = plt.get_cmap('Set1').colors

# X-axis positions for datasets
x = np.arange(len(datasets))

# Width of the bars
width = 0.35

# Params
plt.rcParams.update({
    'font.size': 20,               # General font size
    'axes.titlesize': 20,          # Axes title font size
    'axes.labelsize': 20,          # Axes labels font size
    'xtick.labelsize': 20,         # X-axis tick labels font size
    'ytick.labelsize': 20,         # Y-axis tick labels font size
    'legend.fontsize': 16,         # Legend font size
    # 'figure.titlesize': 16         # Figure title font size
})

# Plotting Runtime Comparison
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, inference_runtimes, width, label='Inference', hatch='//', edgecolor=colors[0], fill=False)
bars2 = ax.bar(x + width/2, training_runtimes, width, label='Train', hatch='xx', edgecolor=colors[1], fill=False)

# Labels and title
# ax.set_xlabel('Datasets')
ax.set_ylabel('Runtime (s)')
# ax.set_title('Runtime Comparison')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.legend()

root_save_path = '/home/dev/workspace/Homomorphic-HalfFed/figs/'
plt.savefig(root_save_path + "motivation_train_and_inference_runtime" + ".jpeg")
pdf = PdfPages(root_save_path + "motivation_train_and_inference_runtime" + '.pdf')
pdf.savefig(bbox_inches = 'tight', pad_inches = 0)
pdf.close()
plt.close()


# Plotting Communication Comparison
fig, ax = plt.subplots()
bars3 = ax.bar(x - width/2, inference_comm_per_image, width, label='Inference', hatch='//', edgecolor=colors[0], fill=False)
bars4 = ax.bar(x + width/2, training_comm_per_image, width, label='Train', hatch='xx', edgecolor=colors[1], fill=False)

# Labels and title
# ax.set_xlabel('Datasets')
ax.set_ylabel('Comm. (MB/img)')
# ax.set_title('Communication Comparison')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper left')

root_save_path = '/home/dev/workspace/Homomorphic-HalfFed/figs/'
plt.savefig(root_save_path + "motivation_train_and_inference_comm" + ".jpeg")
pdf = PdfPages(root_save_path + "motivation_train_and_inference_comm" + '.pdf')
pdf.savefig(bbox_inches = 'tight', pad_inches = 0)
pdf.close()
plt.close()

