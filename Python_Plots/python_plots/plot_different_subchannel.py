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

file_string = 'Different_subcarriers'
rows_as_group = ['20', '40', '80', '160', '234']
columns_as_bars = ["Setup 1", "Setup 2", "Setup 3"]





data_file = '../' + file_string + '.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'


data = np.loadtxt(data_file)
data = data[:, 1:]

column_1 = data[:, 0]
column_2 = data[:, 1]
column_3 = data[:, 2]


x = np.arange(len(rows_as_group))
width = 0.2
fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(10)

rects1 = ax.bar(x - width, column_1, width, color='lavender', hatch='/', edgecolor="black", linewidth=2)
rects2 = ax.bar(x, column_2, width, color='lightcyan', hatch=' \ ', edgecolor="black", linewidth=2)
rects3 = ax.bar(x + width, column_3, width, color='lavenderblush', hatch='//', edgecolor="black", linewidth=2)

ax.set_ylabel('Accuracy (\%)', fontsize=25)
ax.set_xlabel('Number of sub-channel', fontsize=25)

ax.set_xticks(x)
ax.set_xticklabels(rows_as_group, fontsize=25)
ax.set_ylim(70, 110)
ax.set_yticks([70, 80, 90, 100, 110], fontsize=25)

ax.legend([rects1, rects2, rects3], columns_as_bars, loc='upper center', ncol=3, fontsize= 25)




plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(fig_pdf_file, dpi=300, format='pdf')
plt.savefig(fig_eps_file, dpi=300, format='eps')

plt.show()
print('1')