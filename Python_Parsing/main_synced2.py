from picoscenes import Picoscenes
import numpy as np
import os
import glob


directory = '/mnt/HDD2/Channel_Sensing_Raw_Data/Experiments_Anechoic_1/location_2/Injector_1/Channel_33/160MHz/'
directory_sync = directory + 'Synced/'
extension = '*.csi'
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
    print(np.shape(globals()[seq_array]))
    np.save(saved_file, globals()[seq_array])

if __name__ == "__main__":
    files = load_files(directory, extension)
    for file in files:
        process_file(file, directory, directory_sync)
