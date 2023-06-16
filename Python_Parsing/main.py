from picoscenes import  Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

array_names =['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09']

directory = '/mnt/SSD2/INFOCOM_23/Data1/Captures_315/'
extension = '*.csi'  # Replace with the desired file extension

# Create the search pattern
search_pattern = os.path.join(directory, extension)

# Use glob to find files matching the pattern
files = glob.glob(search_pattern)

for file in files:
    seq_array=file[-7:-4]
    seq_set= file[-7:-4] + "_set"


    globals()[seq_array]=[]
    A=[]

    print(file)
    increment = 80000
    group_size = 4096
    # stands for the first frame of csi frames

    frames = Picoscenes(file)
    for i in range(12288):
        Sequence = frames.raw[i].get("StandardHeader").get("Sequence")
        np.array(A.append((i, Sequence)))

    for i in range(0, len(A), group_size):
        indices = np.arange(i, i + group_size)
        A_indices = np.array(A)
        A_indices[indices, 1] += increment
        A = A_indices.tolist()
        increment *= 2

    globals()[seq_array] = A
    globals()[seq_set] = {t[1] for t in globals()[seq_array]}


synced_sequence = A01_set.intersection(A02_set, A03_set, A04_set, A05_set, A06_set, A07_set, A08_set, A09_set, B01_set, B02_set, B03_set, B04_set, B05_set, B06_set, B07_set, B08_set, B09_set )
synced_seq = np.array(list(synced_sequence))

#common_tuples = A_set.intersection(C_set, D_set, {(x, y) for (x, y) in A if y in B})

def find_corresponding_first_element(array, synced_array):
    return [tuple_element[0] for tuple_element in array if tuple_element[1] in synced_array]


for name in array_names:
    result=[]
    save_name = directory + name + ".npy"

    corresponding_element= find_corresponding_first_element(globals()[name], synced_seq)
    corresponding_element = list(set(corresponding_element))
    np.save(save_name, corresponding_element)
