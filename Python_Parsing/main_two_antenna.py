from picoscenes import Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


directory = '../Data_two_antenna/Pannel_A/'
extension = '*.csi'  # Replace with the desired file extension

# Create the search pattern
search_pattern = os.path.join(directory, extension)

# Use glob to find files matching the pattern
files = glob.glob(search_pattern)

for file in files:
    seq_array_one = file[-6:-5]
    seq_array_two = file[-5:-4]
    globals()[seq_array_one] = []
    globals()[seq_array_two] = []

    A = []
    frames = Picoscenes(file)

    CSI_base_one= np.array(frames.raw[0].get("CSI").get("CSI"))[:2025]
    CSI_base_two= np.array(frames.raw[0].get("CSI").get("CSI"))[2025:]
    globals()[seq_array_one].append(CSI_base_one)
    globals()[seq_array_two].append(CSI_base_two)
    for i in range(1, 500):
        saved_file_one = directory + 'Synced/' + file[-6:-5] + ".npy"
        saved_file_two = directory + 'Synced/' + file[-5:-4] + ".npy"
        #print(saved_file_one)
        #print(saved_file_two)

        numTones = frames.raw[i].get("CSI").get("numTones")
        CSI_one = np.array(frames.raw[i].get("CSI").get("CSI"))[:numTones]
        CSI_two = np.array(frames.raw[i].get("CSI").get("CSI"))[numTones:]

        # CSI_next_one = np.array(frames.raw[i+1].get("CSI").get("CSI"))[:numTones]
        # CSI_next_two = np.array(frames.raw[i+1].get("CSI").get("CSI"))[numTones:]
        #
        #

        correlation_coefficient_one = np.abs(np.corrcoef(CSI_base_one, CSI_one)[0, 1])
        correlation_coefficient_two = np.abs(np.corrcoef(CSI_base_two, CSI_one)[0, 1])

        if (correlation_coefficient_one) > (correlation_coefficient_two):
            globals()[seq_array_one].append(CSI_one)
            globals()[seq_array_two].append(CSI_two)
            CSI_base_one = CSI_one

        else:
            globals()[seq_array_two].append(CSI_one)
            globals()[seq_array_one].append(CSI_two)
            CSI_base_one = CSI_two

        a = np.array(globals()[seq_array_one])
        b = np.array(globals()[seq_array_two])
        # #similarity_one = cosine(CSI_one, CSI_next_one)
        # similarity_one = cosine_similarity(CSI_one.reshape(1, -1), CSI_next_one.reshape(1, -1))
        # #similarity_two = cosine(CSI_one, CSI_next_two)
        # similarity_two = cosine_similarity(CSI_one.reshape(1, -1), CSI_next_two.reshape(1, -1))

        # Calculate the complex correlation coefficient
        # numerator = np.sum(np.multiply(CSI_one, np.conj(CSI_next_one)))
        # denominator = np.sqrt(np.sum(np.square(np.abs(CSI_one))) * np.sum(np.square(np.abs(CSI_next_one))))
        # similarity_one = numerator / denominator
        #
        #
        # numerator = np.sum(np.multiply(CSI_one, np.conj(CSI_next_two)))
        # denominator = np.sqrt(np.sum(np.square(np.abs(CSI_one))) * np.sum(np.square(np.abs(CSI_next_two))))
        # similarity_two = numerator / denominator

        # print(i)
        # print(correlation_coefficient_one)
        # print(correlation_coefficient_two)

        # print(similarity_one)
        # print(similarity_two)

        # np.array(globals()[seq_array_one].append(CSI_one))
        # np.array(globals()[seq_array_two].append(CSI_two))

    np.save(saved_file_one, globals()[seq_array_one])
    np.save(saved_file_two, globals()[seq_array_two])



csi_magnitude = np.abs(b[20:30, :])
x = np.arange(len(csi_magnitude))


num_packets = 10
num_subcarriers = 2025

# Create a figure and axes
fig, ax = plt.subplots()

# Iterate over each packet
for i in range(num_packets):
    # Plot the CSI magnitude for the current packet
    ax.plot(range(num_subcarriers), csi_magnitude[i], label=f'Packet {i+1}')




# Add labels and title
plt.xlabel('Index')
plt.ylabel('CSI Magnitude')
plt.title('Magnitude of CSI Data')

# Display the plot
plt.show()
print('1')