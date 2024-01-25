# Imports

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import matplotlib as mpl

mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['text.usetex'] = 'true'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
mpl.rcParams['font.size'] = 22
#mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set3.colors)

# Change Params Here:


def add_label1(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=28)



def add_label2(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(1, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=28)


fontsize = 125
# Model names
models = ['1', '2', '3', '4']

# Accuracy and Time data (example values)
#accuracy = [0.85, 0.92, 0.78, 0.88, 0.5]       # Example accuracy values (left axis)
# inference_time = [10, 15, 8, 12, 6]           # Example inference time values
# transmission_time = [1, 4, 0, 0.5, 0.7]           # Example transmission time values
#process_time = [0, 0, 0, 0, 0]               # Example process time values



file_string = 'different_alpha'
data_file = '../' + file_string + '.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'


data = np.loadtxt(data_file)

#data = data[:, 1:]

accuracy = data[:, 0] * 100
data_size = data[:, 1]
# transmission_time = data[:, 2] * 1000
# compression_time = data[:, 2] * 1000

print(accuracy)
print(data_size)


# Create a figure with a single subplot
fig, ax1 = plt.subplots(figsize=(22, 16.3))

# Bar width for accuracy (left axis) and times (right axis)
bar_width = 0.3

# Offset for the bars to separate them
offset = 0.15

# Bar plot for accuracy (left axis)
accuracy_bars = np.arange(len(models))
rects1= ax1.bar(accuracy_bars - offset, accuracy, width=bar_width, color='skyblue', hatch='/', edgecolor="black", linewidth=2,label='$mAP_{50-95}$')
ax1.set_ylabel('$mAP_{50-95} (\%)$', color='b', fontsize=fontsize)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yticks([30,  60, 90], [30,  60, 90], fontsize=fontsize)
ax1.set_ylim([0, 140])  # Set the y-axis range for accuracy (0 to 1)

# Create a second y-axis for times (right axis)
ax2 = ax1.twinx()
time_bars = accuracy_bars + offset
rects2 = ax2.bar(time_bars, data_size, width=bar_width, color='lightcoral', hatch='/', edgecolor="black", linewidth=2, alpha=0.7, label='Channel\noccupation')
# ax2.bar(time_bars, transmission_time, width=bar_width, bottom=infe, color='lightcoral', hatch='\\', edgecolor="black", linewidth=2,label='Tx time')
# ax2.bar(time_bars, compression_time, width=bar_width, bottom=np.array(inference_time) + np.array(transmission_time), color='lightgreen', hatch='//', edgecolor="black", linewidth=2, alpha=0.7, label='Compression time')

ax2.set_ylabel('Channel occupa-\n-tion/ROI(MB)', color='r', fontsize=fontsize)
ax2.tick_params(axis='y', labelcolor='r')
#ax2.set_ylim([0, max(sum(zip(inference_time, transmission_time, compression_time), ())) + 10])  # Set the y-axis range for times
ax2.set_ylim([0, 22])
ax2.set_yticks([0,  10, 20], [0,  10, 20], fontsize=fontsize)
ax2.yaxis.set_tick_params(which='both', labelright=True)  # Adjust the pad parameter as needed


# Set the x-axis ticks and labels
ax1.set_xticks(accuracy_bars)
ax1.set_xticklabels(models, fontsize=fontsize)
ax1.set_xlabel(r'ROI multiplying factor, $\alpha$', fontsize=fontsize)

# Increase y-axis tick font size
ax1.tick_params(axis='both', which='both', labelsize=fontsize)
ax2.tick_params(axis='both', which='both', labelsize=fontsize)


# add_label1(rects1)
# add_label2(rects2)

ax1.grid(axis='y', linestyle='--', linewidth=0.5)

# Adding legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper center', ncol=1, fontsize=fontsize-15, framealpha=0.0, bbox_to_anchor=(0.43, 0.98), bbox_transform=plt.gcf().transFigure, columnspacing=-1.5)

# Title and display the plot
#plt.title('Accuracy and Time (Inference, Transmission & Process) for Different Models', fontsize=15)

plt.tight_layout()
#
plt.savefig(fig_pdf_file, dpi=300, format='pdf')
plt.savefig(fig_eps_file, dpi=300, format='eps')

plt.show()
