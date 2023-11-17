import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['Model 1', 'Model 2', 'Model 3', 'Model 4']

# Accuracy and Inference Time data (example values)
accuracy = [0.85, 0.92, 0.78, 0.88]  # Example accuracy values (left axis)
inference_time = [10, 15, 8, 12]     # Example inference time values (right axis)

# Create a figure with a single subplot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Bar width for accuracy (left axis) and inference time (right axis)
bar_width = 0.3

# Offset for the bars to separate them
offset = 0.15

# Bar plot for accuracy (left axis)      #ffe4e1
accuracy_bars = np.arange(len(models))
ax1.bar(accuracy_bars - offset, accuracy, width=bar_width, color='lightcyan', hatch='/', edgecolor="black", linewidth=2, alpha=0.7, label='Accuracy')
#ax1.set_xlabel('Models', fontsize=25)
ax1.set_ylabel('Accuracy', color='b', fontsize=25)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim([0, 1.2])  # Set the y-axis range for accuracy (0 to 1)

# Create a second y-axis for inference time (right axis)
ax2 = ax1.twinx()
inference_time_bars = accuracy_bars + offset
ax2.bar(inference_time_bars, inference_time, width=bar_width, color='#D8BFD8', hatch=' \ ', edgecolor="black", linewidth=2, alpha=0.7, label='Inference Time')
ax2.set_ylabel('Inference Time (ms)', color='r', fontsize=25)
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim([5, max(inference_time) + 20])  # Set the y-axis range for inference time
ax2.set_yticks([10, 15, 20, 25, 30], fontsize=55)

# Set the x-axis ticks and labels
ax1.set_xticks(accuracy_bars, fontsize=25)
ax1.set_xticklabels(models, fontsize=25)
ax1.set_yticks([0.2, 0.40, 0.6, 0.8, 1], fontsize=55)

# Increase y-axis tick font size
ax1.tick_params(axis='both', which='both', labelsize=25)
ax2.tick_params(axis='both', which='both', labelsize=25)

plt.grid(axis='y', linestyle='--', linewidth=0.5)

# Adding legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper center', ncol=3, fontsize= 25)

# Title and display the plot
plt.title('Accuracy and Inference Time for Different Models', fontsize=25)
plt.tight_layout()
plt.show()
