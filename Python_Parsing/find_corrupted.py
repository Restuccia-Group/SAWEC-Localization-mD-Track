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

BW = 20
directory = '/mnt/HDD1/Channel_Sensing_Raw_Data/Experiments_1_Classroom/location_2/Injector_1/Channel_1/' + str(BW) + 'MHz/Synced/'
extension = '*.npy'  # Replace with the desired file extension

if BW == 20:
    subcarrier = 245
elif BW == 40:
    subcarrier = 489
elif BW == 80:
    subcarrier = 1001
else:
    subcarrier = 2025



# Create the search pattern
search_pattern = os.path.join(directory, extension)
search_pattern_Antenna_Separated = os.path.join(directory, 'Antenna_Separated', extension)

# Use glob to find files matching the pattern
files = glob.glob(search_pattern)
files_Antenna_Separated = glob.glob(search_pattern_Antenna_Separated)

Antenna_Sync = []

for file in files:

    CSI = np.load(file)

    for i in range(len(CSI[:-1])):
        saved_file_one = directory + 'Antenna_Separated/' + file[-7:-4] + '_1' + ".npy"
        saved_file_two = directory + 'Antenna_Separated/' + file[-7:-4] + '_2' + ".npy"

        CSI_base_one = CSI[i, :subcarrier]
        CSI_base_two = CSI[i, subcarrier:]

        CSI_one = CSI[i+1, :subcarrier]
        CSI_two = CSI[i+1, subcarrier:]


        correlation_coefficient_two = np.abs(cosine_similarity(np.abs(CSI_base_two), np.abs(CSI_two)))
        correlation_coefficient_one = np.abs(cosine_similarity(np.abs(CSI_base_two), np.abs(CSI_one)))

        if (abs(correlation_coefficient_two - correlation_coefficient_one)) < 0.005:
            # print(file)
            #print(i)
            Antenna_Sync.append(i)
    print('1')

Unique_Antenna_Sync = list(set(Antenna_Sync))
print('1')

for file in files:
    CSI = np.load(file)
    CSI = np.delete(CSI, Unique_Antenna_Sync, axis=0)
    print(np.shape(CSI))
    np.save(file, CSI)

print('1')
            # print(correlation_coefficient_two)
            # print(correlation_coefficient_one)


    #
    # print(np.shape(globals()[seq_array_one]))
    # print(np.shape(globals()[seq_array_two]))
    #
    # np.save(saved_file_one, globals()[seq_array_one])
    # np.save(saved_file_two, globals()[seq_array_two])
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





# for i in range(len(Antenna_Sync)):
#     for x in files_Antenna_Separated:
#         if x[-9:-6] != (Antenna_Sync[i][0][-7:-4]):
#             CSI = np.load(x)
#             print(i)
#             CSI = np.delete(CSI, Antenna_Sync[i][1], axis=0)
#             saved_name = x
#             np.save(saved_name, CSI)







