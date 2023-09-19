# Imports

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import matplotlib as mpl

# mpl.rcParams['font.serif'] = 'Palatino'
# mpl.rcParams['text.usetex'] = 'true'
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
mpl.rcParams['font.size'] = 22
#mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set3.colors)

# Change Params Here:

file_string = 'expression_detection'
rows_as_group = ['32x32', '128x128', '256x256', '512x512', '1024x1024']
columns_as_bars = ['legend']

# Data
labels = ['Bar 1', 'Bar 2', 'Bar 3']
values = [93.01, 93.54, 96.55]



data_file = '../' + file_string + '.txt'
#data_file = '/home/foysal/Github/Beamforming_Feedback_Extraction_IEEE802.11ax/Use_Case/BeamSense_angles/Different_Setup/Plots/expression_detection.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'


data = np.loadtxt(data_file)
data = data[:, 1:]

column_1 = data[:, 0]
# column_1 = data[:, 1]
# column_1 = data[:, 2]


x = np.arange(len(rows_as_group))
width = 0.2
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(13)

rects1 = ax.bar(rows_as_group, column_1, color='lavender', hatch='/', edgecolor="black", linewidth=2, width=0.3)

ax.set_ylabel('Accuracy (%)', fontsize=35)
ax.set_xlabel('Image Resolution', fontsize=35)

ax.set_xticks(x)
#plt.xticks(rotation=25)
ax.set_xticklabels(rows_as_group, fontsize=30)
ax.set_ylim(0, 110)

ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels([20, 40, 60, 80, 100], fontsize=30)

#ax.legend([rects1], columns_as_bars, loc='upper center', ncol=3, fontsize= 25)




plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(fig_pdf_file, dpi=300, format='pdf')
plt.savefig(fig_eps_file, dpi=300, format='eps')

plt.show()
print('1')