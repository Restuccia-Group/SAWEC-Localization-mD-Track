from picoscenes import Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity(signal1, signal2):
    dot_product = np.vdot(signal1, signal2)
    norm_signal1 = np.linalg.norm(signal1)
    norm_signal2 = np.linalg.norm(signal2)

    similarity = dot_product / (norm_signal1 * norm_signal2)
    return similarity

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


        # correlation_coefficient_two = np.abs(np.corrcoef(CSI_base_two, CSI_two)[0, 1])
        # correlation_coefficient_one = np.abs(np.corrcoef(CSI_base_two, CSI_one)[0, 1])

        correlation_coefficient_two = np.abs(cosine_similarity(np.abs(CSI_base_two), np.abs(CSI_two)))
        correlation_coefficient_one = np.abs(cosine_similarity(np.abs(CSI_base_two), np.abs(CSI_one)))
9


        if (correlation_coefficient_two) > (correlation_coefficient_one):
            globals()[seq_array_one].append(CSI_one)
            globals()[seq_array_two].append(CSI_two)
            CSI_base_two = CSI_two

        else:
            globals()[seq_array_two].append(CSI_one)
            globals()[seq_array_one].append(CSI_two)
            CSI_base_two = CSI_one



    ### PLot
    # a = np.array(globals()[seq_array_one])
    # b = np.array(globals()[seq_array_two])
    #
    # a = abs(a[10:500, :])
    # num_packets, num_subcarriers = a.shape
    #
    # # Create a figure and axes
    # fig, ax = plt.subplots()
    #
    # # Iterate over each packet
    # for i in range(num_packets):
    #     # Plot the CSI magnitude for the current packet
    #     ax.plot(range(num_subcarriers), a[i], label=f'Packet {i + 1}')
    # # Add labels and title
    # ax.set_xlabel('Subcarrier Index')
    # ax.set_ylabel('CSI Magnitude')
    # ax.set_title('CSI Magnitude for Each Packet')
    # plt.show()


    np.save(saved_file_one, globals()[seq_array_one])
    np.save(saved_file_two, globals()[seq_array_two])
