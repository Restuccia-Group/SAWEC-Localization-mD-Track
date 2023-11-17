import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['1/32', '1/16', '1/8', '1/4', '1/2']

# Accuracy and Time data (example values)
#accuracy = [0.85, 0.92, 0.78, 0.88, 0.5]       # Example accuracy values (left axis)
# inference_time = [10, 15, 8, 12, 6]           # Example inference time values
# transmission_time = [1, 4, 0, 0.5, 0.7]           # Example transmission time values
#process_time = [0, 0, 0, 0, 0]               # Example process time values



file_string = 'expression_detection_compression'
data_file = '../' + file_string + '.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'


data = np.loadtxt(data_file)

data = data[:, 1:]

accuracy = data[:, 0]
inference_time = data[:, 1] * 1000
transmission_time = data[:, 2] * 1000
compression_time = data[:, 2] * 1000

print(accuracy)


# Create a figure with a single subplot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar width for accuracy (left axis) and times (right axis)
bar_width = 0.3

# Offset for the bars to separate them
offset = 0.15

# Bar plot for accuracy (left axis)
accuracy_bars = np.arange(len(models))
ax1.bar(accuracy_bars - offset, accuracy, width=bar_width, color='lightcyan', hatch='/', edgecolor="black", linewidth=2,label='Accuracy')
ax1.set_ylabel('Accuracy', color='b', fontsize=30)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yticks([20, 40, 60, 80, 100], [20, 40, 60, 80, 100], fontsize=25)
ax1.set_ylim([0, 140])  # Set the y-axis range for accuracy (0 to 1)

# Create a second y-axis for times (right axis)
ax2 = ax1.twinx()
time_bars = accuracy_bars + offset
ax2.bar(time_bars, inference_time, width=bar_width, color='skyblue', hatch='/', edgecolor="black", linewidth=2, alpha=0.7, label='Inference time')
ax2.bar(time_bars, transmission_time, width=bar_width, bottom=inference_time, color='lightcoral', hatch='\\', edgecolor="black", linewidth=2,label='Tx ime')
ax2.bar(time_bars, compression_time, width=bar_width, bottom=np.array(inference_time) + np.array(transmission_time), color='lightgreen', hatch='//', edgecolor="black", linewidth=2, alpha=0.7, label='Compression time')

ax2.set_ylabel('End-to-end latency (ms)', color='r', fontsize=30)
ax2.tick_params(axis='y', labelcolor='r')
#ax2.set_ylim([0, max(sum(zip(inference_time, transmission_time, compression_time), ())) + 10])  # Set the y-axis range for times
ax2.set_ylim([95, 115])

# Set the x-axis ticks and labels
ax1.set_xticks(accuracy_bars)
ax1.set_xticklabels(models, fontsize=25)
ax1.set_xlabel('Compression ratio', fontsize=25)

# Increase y-axis tick font size
ax1.tick_params(axis='both', which='both', labelsize=25)
ax2.tick_params(axis='both', which='both', labelsize=25)

ax1.grid(axis='y', linestyle='--', linewidth=0.5)

# Adding legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper center', ncol=2, fontsize=25)

# Title and display the plot
#plt.title('Accuracy and Time (Inference, Transmission & Process) for Different Models', fontsize=15)
plt.tight_layout()


plt.savefig(fig_pdf_file, dpi=300, format='pdf')
plt.savefig(fig_eps_file, dpi=300, format='eps')
plt.show()
