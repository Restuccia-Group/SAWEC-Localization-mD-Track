

# Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import matplotlib as mpl

mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['text.usetex'] = 'true'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
mpl.rcParams['font.size'] = 22

fontsize = 55
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=26)

# Change Params Here:
file_string = 'average_ete_latency_isac-ec'
rows_as_group = ['Compressed\n(1/8)', 'Resized\n(1/2)', 'Original\n(10K)']
columns_as_bars = ["ISAC-EC", "YolactACOS", "EdgeDuet"]

data_file = '../' + file_string + '.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'

data = np.loadtxt(data_file)
data = data[:, :]

column_1 = data[:, 0]
column_2 = data[:, 1]
column_3 = data[:, 2]

x = np.arange(len(rows_as_group))
width = 0.15
fig, ax = plt.subplots()
fig.set_figheight(9)
fig.set_figwidth(16)

rects1 = ax.bar(x - width, column_1, width, align='edge', color='skyblue', hatch='/', edgecolor="black", linewidth=2)
rects2 = ax.bar(x, column_2, width, align='edge', color='lightcyan', hatch=' \ ', edgecolor="black", linewidth=2)
rects3 = ax.bar(x + width , column_3, width, align='edge', color='lightcoral', hatch=' \ ', edgecolor="black", linewidth=2)

# add_labels(rects1)
# add_labels(rects2)
# add_labels(rects3)

ax.set_ylabel('End-to-end latency (ms)', fontsize=fontsize)
ax.set_xticks(x + width/2)
ax.set_xticklabels(rows_as_group, fontsize=fontsize)
ax.set_ylim(1, 30000)  # Adjusted the lower limit to avoid issues with log scale
ax.set_yscale('log')  # Set y-axis to a logarithmic scale
ax.set_yticks([1, 10, 100, 1000, 10000], [1, 10, 100, 1000, 10000], fontsize=fontsize)  # Adjusted the y-axis ticks for log scale

ax.legend([rects1, rects2, rects3], columns_as_bars, loc='upper center', ncol=2, fontsize=fontsize-5, framealpha=0.0)

plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(fig_pdf_file, dpi=300, format='pdf')
#plt.savefig(fig_eps_file, dpi=300, format='eps')

plt.show()
print('1')
