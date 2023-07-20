import numpy as np


A = ['A05', 'A04', 'A03', 'A02', 'A01']


for file_name in A:
    csi_file = '../Data1/Captures_0/Synced/' + file_name + '.npy'
    globals()[file_name] = np.load(csi_file)
    globals()[file_name]= globals()[file_name][:11000]

AA = [A05, A04, A03, A02, A01]

signal_complete = np.stack(AA, axis=2)


# B = np.random.rand(100, 25)
# C = np.random.rand(100, 25)
#
#
# signal_complete= np.stack((B,C), axis=2)

print('1')

