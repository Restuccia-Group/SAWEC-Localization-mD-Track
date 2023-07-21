from picoscenes import  Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#array_names =['A01']
BW = 40
directory = '/mnt/HDD1/Channel_Sensing_Raw_Data/Experiments_1_Classroom/calibration/Injector_1/Channel_1/' + str(BW) + 'MHz/Synced/'
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

# Use glob to find files matching the pattern
files = glob.glob(search_pattern)

for file in files:
    seq_array_one = file[-7:-4] + '_1'
    seq_array_two = file[-7:-4] + '_2'

    CSI = np.load(file)

    globals()[seq_array_one] = []
    globals()[seq_array_two] = []
    A = []

    for i in range(len(CSI[:])):
        saved_file_one = directory + 'Antenna_Separated/' + file[-7:-4] + '_1' + ".npy"
        saved_file_two = directory + 'Antenna_Separated/' + file[-7:-4] + '_2' + ".npy"

        CSI_one = CSI[i, :subcarrier]
        CSI_two = CSI[i, subcarrier:]
        np.array(globals()[seq_array_one].append(CSI_one))
        np.array(globals()[seq_array_two].append(CSI_two))


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(globals()[seq_array_one][:])):
        ax[0].plot(abs(globals()[seq_array_one][i]), label='Antenna 1')
        ax[1].plot(abs(globals()[seq_array_two][i]), label='Antenna 2')
        #plt.plot(abs(globals()[seq_array_two][i]))
    plt.show()


    np.save(saved_file_one, globals()[seq_array_one])
    np.save(saved_file_two, globals()[seq_array_two])

