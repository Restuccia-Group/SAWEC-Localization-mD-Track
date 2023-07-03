import numpy as np
from scipy.linalg import eigh

# Antenna spacing (in meters)
antenna_spacing = 0.05

# Center frequency (in MHz)
center_frequency = 5250

# Define the dimensions of the data
num_snapshots = 500  # Number of snapshots or samples
num_antennas = 5  # Number of antennas
num_subcarriers = 242  # Number of subcarriers

# Generate complex-valued random data for the Channel State Information (CSI)
csi_data = np.random.rand(num_snapshots, num_antennas, num_subcarriers) + \
           1j * np.random.rand(num_snapshots, num_antennas, num_subcarriers)

# Reshape the CSI data to ensure a square covariance matrix
csi_reshaped = np.reshape(csi_data, (num_snapshots, num_antennas * num_subcarriers))

# Compute the covariance matrix
covariance_matrix = np.matmul(csi_reshaped.transpose(), csi_reshaped.conj()) / num_snapshots

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = eigh(covariance_matrix)

# Estimate the number of sources (AoA)
noise_eigenvalues = eigenvalues[:-1]
threshold = np.mean(noise_eigenvalues) + 3 * np.std(noise_eigenvalues)
num_sources = np.sum(eigenvalues > threshold)

# Compute the spatial spectrum
spatial_spectrum = np.abs(np.matmul(csi_reshaped, eigenvectors[:, -num_sources:]))




# Convert frequencies to wavelengths
wavelength = 3e8 / (center_frequency * 1e6)

# Compute the angle of arrival (AoA) for each subcarrier
aoa = np.arcsin(np.arange(num_subcarriers) * wavelength / (antenna_spacing * (num_antennas - 1)))

# Find the peaks in the spatial spectrum for each subcarrier
peaks = np.argmax(spatial_spectrum, axis=1)

# Print the estimated AoA for each subcarrier
for subcarrier, peak in enumerate(peaks):
    print("Subcarrier:", subcarrier)
    print("Estimated AoA:", np.degrees(aoa[peak]))