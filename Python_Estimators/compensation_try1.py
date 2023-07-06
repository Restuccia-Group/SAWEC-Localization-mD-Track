import numpy as np
import matplotlib.pyplot as plt
import cmath
a = np.load('../Data6/Pannel_A/Synced/Antenna_Separated/A01_2.npy')
b = np.load('../Data6/Pannel_A/Synced/Antenna_Separated/A03_2.npy')

subtracted = a - b

plt.figure()
for i in range(len(subtracted[100:101])):
    plt.plot(abs(subtracted[i]))

plt.figure()
for i in range(len(a[100:101])):
    plt.plot(abs(a[i]))
    plt.plot(abs(b[i]))
plt.show()


# plt.figure()
# # Calculate phase values
# phase_values = np.angle(subtracted)
#
# # Unwrap phase values
# unwrapped_phase_values = np.unwrap(phase_values, axis=1)
#
#
# # Plot unwrapped CSI phases
# for i in range(len(unwrapped_phase_values[:])):
#     plt.plot(unwrapped_phase_values[i])
#
# # Set plot title and labels
# plt.title('Unwrapped CSI Phases')
# plt.xlabel('Subcarrier Index')
# plt.ylabel('Phase (radians)')

# Display the plot
plt.show()


print('1')