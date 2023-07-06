from picoscenes import Picoscenes
import numpy as np
import os
import glob

def load_files(directory, extension):
    search_pattern = os.path.join(directory, extension)
    return glob.glob(search_pattern)

def process_file(file, directory, directory_sync):
    seq_array = file[-7:-4]
    synced_file = os.path.join(directory, file[-7:-4] + ".npy")
    saved_file = os.path.join(directory_sync, file[-7:-4] + ".npy")
    S = np.load(synced_file)
    globals()[seq_array] = []
    frames = Picoscenes(file)
    for i in S:
        numTones = frames.raw[i].get("CSI").get("numTones")
        CSI = np.array(frames.raw[i].get("CSI").get("CSI"))
        np.array(globals()[seq_array].append(CSI))
    np.save(saved_file, globals()[seq_array])

if __name__ == "__main__":
    directory = '../Data7/'
    directory_sync = '../Data7/Synced/'
    extension = '*.csi'
    files = load_files(directory, extension)
    for file in files:
        process_file(file, directory, directory_sync)
