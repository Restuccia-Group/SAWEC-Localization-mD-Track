from picoscenes import Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import time

def cosine_similarity(signal1, signal2):
    dot_product = np.vdot(signal1, signal2)
    norm_signal1 = np.linalg.norm(signal1)
    norm_signal2 = np.linalg.norm(signal2)

    similarity = dot_product / (norm_signal1 * norm_signal2)
    return similarity

BW = 160
directory = '/mnt/HDD1/Channel_Sensing_Raw_Data/Experiments_1_Classroom/location_2/Injector_2/Channel_33/' + str(BW) + 'MHz/Synced/'
#directory = '/mnt/HDD1/Channel_Sensing_Raw_Data/11n_calibration_setup4/Reference/Synced/'
extension = '*.npy'  # Replace with the desired file extension

if BW == 20:
    subcarrier = 245
elif BW == 40:
    subcarrier = 489
elif BW == 80:
    subcarrier = 1001
else:
    subcarrier = 2025



A01 =A02 =A03 =[]

# Create the search pattern
search_pattern = os.path.join(directory, extension)
search_pattern_Antenna_Separated = os.path.join(directory, 'Antenna_Separated', extension)

# Use glob to find files matching the pattern
files = glob.glob(search_pattern)
files_Antenna_Separated = glob.glob(search_pattern_Antenna_Separated)

Antenna_Sync = []

for file in files:
    seq_array_one = file[-7:-4] + '_1'
    seq_array_two = file[-7:-4] + '_2'

    globals()[seq_array_one] = []
    globals()[seq_array_two] = []

    CSI = np.load(file)
    CSI_base_one = CSI[0, :subcarrier]
    CSI_base_two = CSI[0, subcarrier:]



    for i in range(len(CSI[:])):
        saved_file_one = directory + 'Antenna_Separated/' + file[-7:-4] + '_1' + ".npy"
        saved_file_two = directory + 'Antenna_Separated/' + file[-7:-4] + '_2' + ".npy"

        CSI_one = CSI[i, :subcarrier]
        CSI_two = CSI[i, subcarrier:]

        correlation_coefficient_two = np.linalg.norm(abs(CSI_base_two) - abs(CSI_two))
        correlation_coefficient_one = np.linalg.norm(abs(CSI_base_two) - abs(CSI_one))
        #
        #
        # correlation_coefficient_two = np.abs(cosine_similarity(np.abs(CSI_base_two), np.abs(CSI_two)))
        # correlation_coefficient_one = np.abs(cosine_similarity(np.abs(CSI_base_two), np.abs(CSI_one)))


        # if (abs(correlation_coefficient_two - correlation_coefficient_one)) < 0.01:
        #
        #     if file[-7:-4] == "A01":
        #         A01.append(i)
        #     elif file[-7:-4] == "A02":
        #         A02.append(i)
        #     elif file[-7:-4] == "A03":
        #         A03.append(i)
        #
        #     continue


        if correlation_coefficient_two < correlation_coefficient_one:
            globals()[seq_array_one].append(CSI_one)
            globals()[seq_array_two].append(CSI_two)
            CSI_base_two = CSI_two
            #print('Okay')


        else:
            globals()[seq_array_two].append(CSI_one)
            globals()[seq_array_one].append(CSI_two)
            CSI_base_two = CSI_one
            #print('Switch')

        # globals()[seq_array_two].append(CSI_two)
        # globals()[seq_array_one].append(CSI_one)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(globals()[seq_array_one][500:800])):
        ax[0].plot(abs(globals()[seq_array_one][i]), label='Antenna 1')
        ax[1].plot(abs(globals()[seq_array_two][i]), label='Antenna 2')
        ax[0].set_title('Antenna One')
        ax[1].set_title('Antenna Two')
        #plt.plot(abs(globals()[seq_array_two][i]))
    plt.show()

    print('1')


    # plt.figure(1)
    # for i in range(len(globals()[seq_array_one][:])):
    #     plt.plot(abs(globals()[seq_array_one][i]))
    #     #plt.plot(abs(globals()[seq_array_two][i]))
    # plt.show()
    #
    # plt.figure(2)
    # for i in range(len(globals()[seq_array_two][:])):
    #     plt.plot(abs(globals()[seq_array_two][i]))
    #     #plt.plot(abs(globals()[seq_array_two][i]))
    # plt.show()

    print(np.shape(globals()[seq_array_one]))
    print(np.shape(globals()[seq_array_two]))

    np.save(saved_file_one, globals()[seq_array_one])
    np.save(saved_file_two, globals()[seq_array_two])








# for i in range(len(Antenna_Sync)):
#     for x in files_Antenna_Separated:
#         if x[-9:-6] != (Antenna_Sync[i][0][-7:-4]):
#             CSI = np.load(x)
#             print(i)
#             CSI = np.delete(CSI, Antenna_Sync[i][1], axis=0)
#             saved_name = x
#             np.save(saved_name, CSI)



#
# for file in files_Antenna_Separated:
#     print('step3')
#     A = []
#     CSI = np.load(file)
#     for index, value in Antenna_Sync:
#         if (index[-7:-4]) != (file[-9:-6]):
#             A = value
#             # Delete the exact index from the CSI array
#     CSI_new = np.delete(CSI, A, axis=0)
#     print(np.shape(CSI_new))
#     np.save(file, CSI_new)


