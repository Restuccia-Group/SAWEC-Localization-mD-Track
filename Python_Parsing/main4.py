from picoscenes import Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

array_names = ['A01', 'A02', 'A03']
directory = '/mnt/HDD1/Channel_Sensing_Raw_Data/Experiments_1_Classroom/location_2/Injector_2/Channel_33/160MHz/'
extension = '*.csi'

def load_files(directory, extension):
    search_pattern = os.path.join(directory, extension)
    return glob.glob(search_pattern)

def process_file(file):
    seq_array = file[-7:-4]
    seq_set = file[-7:-4] + "_set"
    globals()[seq_array] = []
    A = []
    seq = []
    print(file)
    increment = 80000
    group_size = 4096
    frames = Picoscenes(file)

    for i in range(frames.count):
        sequence = frames.raw[i].get("StandardHeader").get("Sequence")
        np.array(A.append((i, sequence)))
    seq = [tuple[1] for tuple in A]
    lower_value = seq[0]
    a = 80000
    for i in range(len(seq)):
        if seq[i] < lower_value:
            lower_value = seq[i]
            a *= 2
            seq[i] += a
        else:
            lower_value = seq[i]
            seq[i] += a
    globals()[seq_array] = [(x[0], seq[i]) for i, x in enumerate(A)]
    globals()[seq_set] = {t[1] for t in globals()[seq_array]}

def find_synced_sequence(array_names):
    synced_sequence = set(globals()[array_names[0] + "_set"])
    for name in array_names[1:]:
        synced_sequence &= set(globals()[name + "_set"])
    return synced_sequence

def find_corresponding_first_element(array, synced_array):
    return [tuple_element[0] for tuple_element in array if tuple_element[1] in synced_array]

def save_corresponding_elements(directory, array_names, synced_sequence):
    for name in array_names:
        result = []
        save_name = directory + name + ".npy"
        corresponding_element = find_corresponding_first_element(globals()[name], synced_sequence)
        corresponding_element = list(set(corresponding_element))
        np.save(save_name, corresponding_element)

if __name__ == "__main__":
    array_names = array_names
    # directory = directory
    # extension = extension
    files = load_files(directory, extension)
    for file in files:
        process_file(file)
    synced_sequence = find_synced_sequence(array_names)
    synced_seq = np.array(list(synced_sequence))
    print(len(synced_sequence))
    save_corresponding_elements(directory, array_names, synced_seq)
