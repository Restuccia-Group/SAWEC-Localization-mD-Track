from picoscenes import  Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#array_names =['A01']

directory = '../Data4/calibration/A02/'
extension = '*.csi'  # Replace with the desired file extension

# Create the search pattern
search_pattern = os.path.join(directory, extension)

# Use glob to find files matching the pattern
files = glob.glob(search_pattern)

for file in files:
    seq_array = file[-7:-4]

    globals()[seq_array]=[]
    A = []

    frames = Picoscenes(file)
    for i in range(500):
        saved_file = directory + 'Synced/' + file[-7:-4] + ".npy"
        numTones = frames.raw[i].get("CSI").get("numTones")
        CSI = np.array(frames.raw[i].get("CSI").get("CSI"))[numTones:]
        np.array(globals()[seq_array].append(CSI))
    np.save(saved_file, globals()[seq_array])
