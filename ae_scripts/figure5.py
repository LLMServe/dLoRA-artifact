import torch
import time
import numpy as np

lora_rank = 16
hidden_size = 4096
test_cnt = 1000

weight = torch.randn(hidden_size, hidden_size).cuda()
lora_A = torch.randn(hidden_size, lora_rank).cuda()
lora_B = torch.randn(lora_rank, hidden_size).cuda()

def warm_up(x):
    for i in range(test_cnt * 10):
        y = torch.mm(x, weight)
        y_prim = torch.mm(torch.mm(x, lora_A), lora_B)
        torch.cuda.synchronize()

def test_Wx(x: torch.tensor):
    lat = []
    for i in range(test_cnt):
        start = time.time()
        y = torch.mm(x, weight)
        torch.cuda.synchronize()
        end = time.time()
        lat.append(end - start)
    mean_lat = np.mean(lat)
    print("Wx Time: ", mean_lat)
    return mean_lat

def test_BAx(x: torch.tensor):
    lat = []
    for i in range(test_cnt):
        start = time.time()
        y_prim = torch.mm(torch.mm(x, lora_A), lora_B)
        torch.cuda.synchronize()
        end = time.time()
        lat.append(end - start)
    mean_lat = np.mean(lat)
    print("BAx Time: ", mean_lat)
    return mean_lat

if __name__ == "__main__":
    batch_sizes = [2, 4, 6, 8, 10, 12, 14, 16]
    wx_lat = []
    bax_lat = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, hidden_size).cuda()
        warm_up(x)
        wx_lat.append(test_Wx(x) * 1e6)
        bax_lat.append(test_BAx(x) * 1e6)
    print("Wx Time(us):", wx_lat)
    print("BAx time(us):", bax_lat)

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    # constants
    num_subfigs = 1
    num_curves = 2

    # Set font and figure size
    font_size = 38
    plt.rc('font',**{'size': font_size, 'family': 'Arial'})
    plt.rc('pdf',fonttype = 42)

    fig_size = (13, 6)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=False, figsize=fig_size)
    matplotlib.rcParams['xtick.minor.size'] = 4.
    matplotlib.rcParams['xtick.major.size'] = 8.
    matplotlib.rcParams['ytick.major.size'] = 6.
    matplotlib.rcParams['ytick.minor.visible'] = False
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

    colors = ['#0072BD', '#D95319']
    labels = {0: 'Wx',  1: 'BAx'}
    markers = {0: 'o', 1: 's', 2: '^'}
    linestyles = {0: 'solid', 1: '--', 2: 'solid'}

    # x-axis setting
    x_label = 'Batch Size'
    x_ticks = [i for i in range(2, 17, 2)]

    # y-axis setting
    y_label = 'Latency(us)'
    y_ticks = [i for i in range(0, 140, 40)]

    axes.set_xlabel(x_label, labelpad=23)
    axes.set_xlim(left=0.5, right=18)
    axes.get_xaxis().set_tick_params(direction='in', pad=7)
    axes.get_xaxis().set_tick_params(which='minor', direction='in')
    axes.set_xticks(x_ticks)

    axes.set_ylabel(y_label)
    axes.set_ylim(bottom=0, top=1)
    axes.set_yticks(y_ticks)
    axes.get_yaxis().set_tick_params(direction='in', pad=4)
    axes.get_xaxis().set_tick_params(direction='in', pad=4)
    axes.get_yaxis().set_tick_params(direction='in', pad=4)
    axes.get_yaxis().set_tick_params(which='minor', direction='in')

    lines = [[] for i in range(num_curves * 1)]
    lines[0], = axes.plot(batch_sizes, wx_lat, label=labels[0], marker = markers[0], color=colors[0], lw=4, markersize=12, linestyle=linestyles[0],zorder=3)
    lines[1], = axes.plot(batch_sizes, bax_lat, label=labels[1], marker = markers[1], color=colors[1], lw=4, markersize=12, linestyle=linestyles[0],zorder=3)

    fig.legend(handles=[lines[0], lines[1]], handlelength=2.36, 
            ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.05), frameon=False, prop={'size':font_size})

    file_path = './figure5.pdf'
    plt.savefig(file_path, bbox_inches='tight', transparent=True)