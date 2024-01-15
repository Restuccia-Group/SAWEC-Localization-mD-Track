import matplotlib.pyplot as plt
import numpy as np

data = {
    'Compressed': {'Value': 1.034, 'Time': 35.03},
    'Resized': {'Value': 0.544, 'Time': 15.62},
    'No Compressed No Resized': {'Value': 1.323, 'Time': 138.88}
}

names = list(data.keys())
values = [entry['Value'] for entry in data.values()]
times = [entry['Time'] for entry in data.values()]

bar_width = 0.35  # Width of the bars

fig, ax1 = plt.subplots()

# Plot the first column (Value) with a logarithmic scale
bar1 = ax1.bar(np.arange(len(names)), values, bar_width, label='Value', log=True)

# Create a second y-axis to plot the second column (Time)
ax2 = ax1.twinx()
ax2.plot(np.arange(len(names)) + bar_width, times, color='r', marker='o', label='Time')

# Add labels, legend, and title
ax1.set_xlabel('Category')
ax1.set_ylabel('Value (log scale)')
ax2.set_ylabel('Time')
ax1.set_xticks(np.arange(len(names)) + bar_width / 2)
ax1.set_xticklabels(names)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Grouped Bar Plot with Logarithmic Scale for Wide Range of Values')

# Show the plot
plt.show()
