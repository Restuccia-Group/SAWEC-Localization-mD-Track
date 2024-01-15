# Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['text.usetex'] = 'true'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
mpl.rcParams['font.size'] = 22

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=30)

# Change Params Here:
file_string = 'ane_ete_latency'
rows_as_group = ['Compressed\n(1/8)', 'Resized\n(1/2)', 'Original\n(10K)']
columns_as_bars = ["SAWEC", "Traditional"]

data_file = '../' + file_string + '.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'

data = np.loadtxt(data_file)
data = data[:, :]


# Extracting data for the first bar (columns 1-3)
bar1_data = data[:, :3]

# Extracting data for the second bar (columns 4-6)
bar2_data = data[:, 4:7]

# Creating positions for the bars
bar_width = 0.2
index = np.arange(len(data))
x = np.arange(len(rows_as_group))


# Defining colors for each stack
colors = ['lightcoral', 'skyblue', 'lavender']

# Adjusting figure size
fig, ax = plt.subplots(figsize=(12.5, 6))

# Plotting the first bar in log scale with three stacks
ax.bar(index, bar1_data[:, 0], bar_width, label='type 1', log=True, color=colors[0], hatch='/', edgecolor="black", linewidth=2)
ax.bar(index, bar1_data[:, 1], bar_width, bottom=bar1_data[:, 0], label='type 2', log=True, color=colors[1], hatch='/', edgecolor="black", linewidth=2)
ax.bar(index, bar1_data[:, 2], bar_width, bottom=bar1_data[:, 0] + bar1_data[:, 1], label='type 3', log=True, color=colors[2], hatch='/', edgecolor="black", linewidth=2)

# Plotting the second bar in log scale with three stacks
ax.bar(index + bar_width, bar2_data[:, 0], bar_width, label='type 1', log=True, color=colors[0], hatch='\\', edgecolor="black", linewidth=2)
ax.bar(index + bar_width, bar2_data[:, 1], bar_width, bottom=bar2_data[:, 0], label='type 2', log=True, color=colors[1], hatch='\\', edgecolor="black", linewidth=2)
ax.bar(index + bar_width, bar2_data[:, 2], bar_width, bottom=bar2_data[:, 0] + bar2_data[:, 1], label='type 3', log=True, color=colors[2], hatch='\\', edgecolor="black", linewidth=2)



ax.set_ylabel('End-to-end latency (ms)', fontsize=35)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(rows_as_group, fontsize=35)
ax.set_ylim(1, 100000)  # Adjusted the lower limit to avoid issues with log scale
ax.set_yscale('log')  # Set y-axis to a logarithmic scale
ax.set_yticks([1, 10, 100, 1000, 10000], [1, 10, 100, 1000, 10000], fontsize=35)  # Adjusted the y-axis ticks for log scale


# Custom legend for white marker and hatch pattern '/'
legend_handles = [Patch(facecolor='white', edgecolor='black', hatch='/', label='SAWEC')]
legend1 = ax.legend(handles=legend_handles, loc='upper right', fontsize=35, ncol=1, bbox_to_anchor=(0.5, 1.05), framealpha=0.0)

# Custom legend for white marker and hatch pattern '\'
legend_handles = [Patch(facecolor='white', edgecolor='black', hatch='\\', label='Traditional')]
legend2 = ax.legend(handles=legend_handles, loc='upper right', fontsize=35, ncol=1, bbox_to_anchor=(.88, 1.05), framealpha=0.0)

# Custom legend for white marker and hatch pattern '\'
legend_handles = [Patch(facecolor='lightcoral', edgecolor='black', hatch='', label='Tx time')]
legend3 = ax.legend(handles=legend_handles, loc='upper right', fontsize=30, ncol=1, bbox_to_anchor=(1, 0.9), framealpha=0.0)

# Custom legend for white marker and hatch pattern '\'
legend_handles = [Patch(facecolor='skyblue', edgecolor='black', hatch='', label='I/O time')]
legend4 = ax.legend(handles=legend_handles, loc='upper right', fontsize=30, ncol=1, bbox_to_anchor=(0.73, 0.9), framealpha=0.0)


# Custom legend for white marker and hatch pattern '\'
legend_handles = [Patch(facecolor='lavender', edgecolor='black', hatch='', label='Inference time')]
legend5 = ax.legend(handles=legend_handles, loc='upper right', fontsize=30, ncol=1, bbox_to_anchor=(0.43, 0.9), framealpha=0.0)

# # Original legend
# legend = ax.legend(loc='upper center', ncol=3, fontsize=30, framealpha=0.0, handlelength=1.5, handletextpad=0.5)
# legend.get_title().set_fontsize('20')  # Adjust legend title font size
# legend.get_title().set_fontweight('bold')  # Optionally, set legend title font weight



# Adding custom legends to the axis
ax.add_artist(legend1)
ax.add_artist(legend2)
ax.add_artist(legend3)
ax.add_artist(legend4)

plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()






# Show the plot
plt.show()