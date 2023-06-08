from picoscenes import  Picoscenes
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

array_names =['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09']

directory = '../Data1/Captures_315/'
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
    seq=[]

    print(file)
    increment = 80000
    group_size = 4096
    # stands for the first frame of csi frames

    frames = Picoscenes(file)
    for i in range(12288):
        sequence = frames.raw[i].get("StandardHeader").get("Sequence")
        np.array(A.append((i, sequence)))

        #seq = [tuple[1] for tuple in A]
        #np.array(seq.append((sequence)))


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


    globals()[seq_array] = [(x[0], seq[i]) for i, x in enumerate (A)]
    globals()[seq_set] = {t[1] for t in globals()[seq_array]}


synced_sequence = set(A01_set) & set(A02_set) & set(A03_set) & set(A04_set) & set(A05_set) & set(A06_set) & set(A07_set) & set(A08_set) & set(A09_set) & set(B01_set) & set(B02_set) & set(B03_set) & set(B04_set) & set(B05_set) & set(B06_set) & set(B07_set) & set(B08_set) & set(B09_set)
synced_seq = np.array(list(synced_sequence))
print(len(synced_sequence))


#common_tuples = A_set.intersection(C_set, D_set, {(x, y) for (x, y) in A if y in B})

def find_corresponding_first_element(array, synced_array):
    return [tuple_element[0] for tuple_element in array if tuple_element[1] in synced_array]


for name in array_names:
    result=[]
    save_name = directory + name + ".npy"

    corresponding_element= find_corresponding_first_element(globals()[name], synced_seq)
    corresponding_element = list(set(corresponding_element))
    print(len(corresponding_element))
    np.save(save_name, corresponding_element)
