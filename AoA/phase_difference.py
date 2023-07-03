import numpy as np
import matplotlib.pyplot as plt

def calculate_aoa(csi_data, wavelength, distance):
    # Extract the phase data from CSI data for two antennas
    phi1 = csi_data[0]  # Phase data for antenna 1
    phi2 = csi_data[1]  # Phase data for antenna 2

    # Compute the phase difference
    delta_phi = np.angle(phi2) - np.angle(phi1)

    # Calculate the Angle of Arrival (AoA)
    aoa = delta_phi * (wavelength / (2 * np.pi * distance))

    return aoa


# A01 = np.load('../Data4/Pannel_A/Synced/A01.npy')
# A02 = np.load('../Data4/Pannel_A/Synced/A01.npy')

# Example usage
csi_data = [
    np.load('../Data3/Pannel_A/Synced/A01.npy'),  # Phase data for antenna 1
    np.load('../Data3/Pannel_A/Synced/A02.npy')  # Phase data for antenna 2
]
wavelength = 0.0571  # Wavelength of the signal
distance = 0.0285 # Distance between the two antennas

aoa = calculate_aoa(csi_data, wavelength, distance)
print("Angle of Arrival (AoA):", aoa)


# Convert AoA values to radians
aoa_rad = np.radians(aoa)

# Create a polar plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

# Plot the AoA
ax.plot(aoa_rad, np.ones_like(aoa), 'ro')

# Set plot properties
ax.set_ylim([0, 2])  # Adjust the y-axis limits as needed
ax.set_yticklabels([])  # Hide y-axis labels
ax.set_xticklabels([])  # Hide x-axis labels
ax.grid(True)

# Show the plot
plt.show()
print('1')
