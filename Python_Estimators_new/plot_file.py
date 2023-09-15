import numpy as np
import time
import pickle
import math as mt
import matplotlib.pyplot as plt
threshold = -2.5  # - 25 dB
import pickle
delta_t = 50
name_base = "refAnt"
num_paths_plot = 5
num_paths_plot = np.int32(num_paths_plot)
start_plot = 2000
end_plot = 2200

delta_t = np.round(delta_t * 1e-11, 11)  # 5E-10  # 1.25e-11
save_dir = '/mnt/HDD2/Channel_Sensing_Raw_Data/Experiments_Anechoic_1/Saved_Files/' + str(delta_t) + '/'


# Construct the file name
name_file = save_dir + 'paths_aoa_list_' + name_base + '.txt'
# Open the file for reading in binary mode ("rb")
with open(name_file, "rb") as fp:
    # Load the data from the file
    aoa_list = pickle.load(fp)


# Construct the file name
name_file = save_dir + 'paths_toa_list_' + name_base + '.txt'
# Open the file for reading in binary mode ("rb")
with open(name_file, "rb") as fp:
    # Load the data from the file
    toa_list = pickle.load(fp)


# Construct the file name
name_file = save_dir + 'paths_amplitude_list_' + name_base + '.txt'
# Open the file for reading in binary mode ("rb")
with open(name_file, "rb") as fp:
    # Load the data from the file
    amplitude_list = pickle.load(fp)




print("Length of amplitude_list:", len(amplitude_list))
print("Length of toa_list:", len(toa_list))
print("Length of aoa_list:", len(aoa_list))

amplitude_list = amplitude_list[start_plot:end_plot]
toa_list = toa_list[start_plot:end_plot]
aoa_list = aoa_list[start_plot:end_plot]

print("Length of amplitude_list:", len(amplitude_list))
print("Length of toa_list:", len(toa_list))
print("Length of aoa_list:", len(aoa_list))


a = 1

def plot_mdtrack_results(amplitude_list, toa_list, aoa_list, num_paths_plot):
    plt.figure()
    vmin = threshold*10
    vmax = 0

    for i in range(len(amplitude_list)):  # number of packets
        print(i)
        sort_idx = np.flip(np.argsort(abs(amplitude_list[i])))
        paths_amplitude_sort = amplitude_list[i][sort_idx]
        paths_power = np.power(np.abs(paths_amplitude_sort), 2)
        paths_power = 10 * np.log10(paths_power / np.amax(np.nan_to_num(paths_power)))  # dB
        paths_toa_sort = toa_list[i][sort_idx]
        paths_aoa_sort = aoa_list[i][sort_idx]


        aoa_array = paths_aoa_sort - paths_aoa_sort[0]
        aoa_array[aoa_array > 0] = aoa_array[aoa_array > 0] + paths_aoa_sort[0]
        aoa_array[aoa_array < 0] = aoa_array[aoa_array < 0] + paths_aoa_sort[0]
        aoa_array[aoa_array > 90] = aoa_array[aoa_array > 90] - 180
        aoa_array[aoa_array < -90] = 180 + aoa_array[aoa_array < -90]

        toa_array = paths_toa_sort - paths_toa_sort[0]

        plt.scatter(toa_array[:num_paths_plot] * 1E9, aoa_array[:num_paths_plot],
                    c=paths_power[:num_paths_plot],
                    marker='o', cmap='Blues', s=12,
                    vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('power [dB]', rotation=90)
    plt.xlabel('ToA [ns]')
    plt.ylabel('AoA [deg]')
    title = 'Antennas idx ' #+ str(antennas_idx_considered)
    plt.title(title)
    plt.grid()
    plt.show()


plot_mdtrack_results(amplitude_list, toa_list, aoa_list, num_paths_plot)