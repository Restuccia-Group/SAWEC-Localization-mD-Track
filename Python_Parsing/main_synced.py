from picoscenes import  Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

directory = '/mnt/SSD2/INFOCOM_23/Data1/Captures_315/'
directory_sync = '/mnt/SSD2/INFOCOM_23/Data1/Captures_315/Synced/'
extension = '*.csi'  # Replace with the desired file extension

# Create the search pattern
search_pattern = os.path.join(directory, extension)

# Use glob to find files matching the pattern
files = glob.glob(search_pattern)

for file in files:
    seq_array = file[-7:-4]
    synced_file = directory + file[-7:-4] + ".npy"
    saved_file = directory_sync + file[-7:-4] + ".npy"
    S = np.load(synced_file)

    globals()[seq_array]=[]

    frames = Picoscenes(file)
    for i in S:
        numTones = frames.raw[i].get("CSI").get("numTones")
        CSI = np.array(frames.raw[i].get("CSI").get("CSI"))[:numTones]

        np.array(globals()[seq_array].append(CSI))
    np.save(saved_file, globals()[seq_array])
