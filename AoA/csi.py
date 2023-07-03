import numpy as np

num_snapshots = 10
num_antennas = 3
num_subcarriers = 2

csi_data = np.random.rand(num_snapshots, num_antennas, num_subcarriers) + \
           1j * np.random.rand(num_snapshots, num_antennas, num_subcarriers)

print("CSI Data:")
print(csi_data)