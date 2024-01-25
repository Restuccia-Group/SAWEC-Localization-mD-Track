import matplotlib.pyplot as plt
import numpy as np

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


fontsize=125
# Change Params Here:

file_string = 'expression_detection_size'
data_file = '../' + file_string + '.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'


data = np.loadtxt(data_file)

# Model names
models = [r'$\frac{1}{16}$', r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$1$']


# Accuracy and Time data (example values)
# accuracy = [0.85, 0.92, 0.78, 0.88]  # Example accuracy values (left axis)
# inference_time = [10, 15, 8, 12]     # Example inference time values
# transmission_time = [5, 7, 4, 6]          # Example process time values



data = data[:, 1:]

accuracy = data[:, 0]
inference_time = data[:, 1] * 1000
transmission_time = data[:, 2] * 1000





# Create a figure with a single subplot
fig, ax1 = plt.subplots(figsize=(20, 15))

# Bar width for accuracy (left axis) and times (right axis)
bar_width = 0.3

# Offset for the bars to separate them
offset = 0.15

# Bar plot for accuracy (left axis)
accuracy_bars = np.arange(len(models))
ax1.bar(accuracy_bars - offset, accuracy, width=bar_width, color='teal', hatch='/', edgecolor="black", linewidth=2,label='Accuracy')
ax1.set_ylabel('Accuracy (\%)', color='b', fontsize=fontsize)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yticks([30,  60, 90], [30,  30, 90], fontsize=fontsize)
ax1.set_ylim([0, 150])  # Set the y-axis range for accuracy (0 to 1)

# Create a second y-axis for times (right axis)
ax2 = ax1.twinx()
time_bars = accuracy_bars + offset
ax2.bar(time_bars, inference_time, width=bar_width, color='skyblue', hatch='/', edgecolor="black", linewidth=2, alpha=0.7, label='Inference time')
ax2.bar(time_bars, transmission_time, width=bar_width, bottom=inference_time, color='lightcoral', hatch='\\', edgecolor="black", linewidth=2,label='Tx time')
ax2.set_ylabel('Total latency(ms)', color='r', fontsize=fontsize-2)
ax2.tick_params(axis='y', labelcolor='r')
#ax2.set_ylim([0, max(max(inference_time), max(transmission_time)) + 0.01])  # Set the y-axis range for times
ax2.set_ylim([90, 121])
# Set the x-axis ticks and labels
ax1.set_xticks(accuracy_bars)
ax1.set_xticklabels(models, fontsize=fontsize)
ax1.set_xlabel('Image downsize ratio', fontsize=fontsize)

# Increase y-axis tick font size
ax1.tick_params(axis='x', which='both', labelsize=fontsize)
ax2.tick_params(axis='both', which='both', labelsize=fontsize)

ax1.grid(axis='y', linestyle='--', linewidth=0.5)

# Adding legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper left', ncol=1, fontsize=fontsize-45, framealpha=0.0, bbox_to_anchor=(0.18, 0.95), bbox_transform=plt.gcf().transFigure, columnspacing=-1.5, handlelength=1)

# Title and display the plot
#plt.title('Accuracy and end-to-end latency for different image size', fontsize=25)
plt.tight_layout()




plt.savefig(fig_pdf_file, dpi=300, format='pdf')
plt.savefig(fig_eps_file, dpi=300, format='eps')

plt.show()
