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
directory = '/mnt/HDD1/Channel_Sensing_Raw_Data/Experiments_1_Classroom/location_1/Injector_2/Channel_33/' + str(BW) + 'MHz/Synced/'
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


        correlation_coefficient_two = np.abs(cosine_similarity(np.abs(CSI_base_two), np.abs(CSI_two)))
        correlation_coefficient_one = np.abs(cosine_similarity(np.abs(CSI_base_two), np.abs(CSI_one)))

        if (abs(correlation_coefficient_two - correlation_coefficient_one)) < 0.02:

            if file[-7:-4] == "A01":
                A01.append(i)
            elif file[-7:-4] == "A02":
                A02.append(i)
            elif file[-7:-4] == "A03":
                A03.append(i)

            continue

        if correlation_coefficient_two > correlation_coefficient_one:
            CSI_base_two = CSI_two

        else:
            CSI_base_two = CSI_one

Delete = np.concatenate((np.array(A01), np.array(A02), np.array(A03)))
Delete_new = list(set(Delete))
print(len(Delete_new))


for file in files:
    CSI = np.load(file)
    CSI = np.delete(CSI, Delete_new, axis=0)
    print(np.shape(CSI))
    np.save(file, CSI)
