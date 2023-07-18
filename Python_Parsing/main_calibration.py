from picoscenes import  Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#array_names =['A01']

directory = '/mnt/HDD1/Channel_Sensing_Raw_Data/Experiments_1_Classroom/calibration/Injector_1/Channel_1/20MHz/'
extension = '*.npy'  # Replace with the desired file extension

# Create the search pattern
search_pattern = os.path.join(directory, extension)

# Use glob to find files matching the pattern
files = glob.glob(search_pattern)

for file in files:
    seq_array_one = file[-7:-4] + '_1'
    seq_array_two = file[-7:-4] + '_2'

    globals()[seq_array_one] = []
    globals()[seq_array_two] = []
    A = []

    frames = Picoscenes(file)
    for i in range(3000):
        saved_file_one = directory + 'Synced' + 'Two_Antenna/' + file[-7:-4] + '_1' + ".npy"
        saved_file_two = directory + 'Synced' + 'Two_Antenna/' + file[-7:-4] + '_2' + ".npy"

        CSI = np.load(file)
        CSI_one = CSI[i, :53]
        CSI_two = CSI[i, 53:]
        np.array(globals()[seq_array_one].append(CSI_one))
        np.array(globals()[seq_array_two].append(CSI_two))
    np.save(saved_file_one, globals()[seq_array_one])
    np.save(saved_file_two, globals()[seq_array_two])

