

# Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import matplotlib as mpl

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

column_1 = data[:, 3]
column_2 = data[:, 7]

x = np.arange(len(rows_as_group))
width = 0.2
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(12.5)

rects1 = ax.bar(x - width, column_1, width, align='edge', color='skyblue', hatch='/', edgecolor="black", linewidth=2)
rects2 = ax.bar(x, column_2, width, align='edge', color='lightcyan', hatch=' \ ', edgecolor="black", linewidth=2)

add_labels(rects1)
add_labels(rects2)

ax.set_ylabel('End-to-end latency (ms)', fontsize=35)
ax.set_xticks(x)
ax.set_xticklabels(rows_as_group, fontsize=35)
ax.set_ylim(1, 100000)  # Adjusted the lower limit to avoid issues with log scale
ax.set_yscale('log')  # Set y-axis to a logarithmic scale
ax.set_yticks([1, 10, 100, 1000, 10000], [1, 10, 100, 1000, 10000], fontsize=35)  # Adjusted the y-axis ticks for log scale

ax.legend([rects1, rects2], columns_as_bars, loc='upper center', ncol=2, fontsize=32, framealpha=0.0)

plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(fig_pdf_file, dpi=300, format='pdf')
plt.savefig(fig_eps_file, dpi=300, format='eps')

plt.show()
print('1')
