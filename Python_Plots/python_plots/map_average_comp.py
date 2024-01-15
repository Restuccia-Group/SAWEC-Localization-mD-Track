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


def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2 , height),
                    xytext=(5, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=30)



# Change Params Here:

file_string = 'map_average_comp'
rows_as_group = ['Compressed\n(1/8)', 'Resized\n(1/2)', 'Original\n(10K)']
columns_as_bars = ["SAWEC", "Entire\nimage", "Partitioned\nimage"]





data_file = '../' + file_string + '.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'


data = np.loadtxt(data_file)
data = data[:, :]

column_1 = data[:, 0] * 100
column_2 = data[:, 1] * 100
column_3 = data[:, 2] * 100


x = np.arange(len(rows_as_group))
width = 0.2
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(12.5)

rects1 = ax.bar(x - width, column_1, width, color='skyblue', hatch='/', edgecolor="black", linewidth=2)
rects2 = ax.bar(x, column_2, width, color='lightcyan', hatch=' \ ', edgecolor="black", linewidth=2)
rects3 = ax.bar(x + width, column_3, width, color='lightcoral', hatch=' \ ', edgecolor="black", linewidth=2)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

ax.set_ylabel('$mAP_{50-95}$ (\%)', fontsize=35)
#ax.set_xlabel('Number of Subcarriers', fontsize=25)
#ax.set_title('Scores by group and gender', fontsize= 25)

ax.set_xticks(x)
ax.set_xticklabels(rows_as_group, fontsize=35)
ax.set_ylim(0, 130)
ax.set_yticks([0, 20, 40, 60, 80], [0, 20, 40, 60, 80], fontsize=35)

ax.legend([rects1, rects2, rects3], columns_as_bars, loc='upper center', ncol=3, fontsize=32, framealpha=0.0)




plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(fig_pdf_file, dpi=300, format='pdf')
plt.savefig(fig_eps_file, dpi=300, format='eps')

plt.show()
print('1')