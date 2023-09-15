
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# Parameters
threshold = -2.5   # - 25 dB
delta_t = 50
name_base = "refAnt"
num_paths_plot = 5
num_paths_plot = np.int32(num_paths_plot)
start_plot = 12000
end_plot = 12100
a = 50  #50
delta_t = np.round(delta_t * 1e-11, 11)  # 5E-10  # 1.25e-11
# Example values (you may need to adjust these)
clustering_epsilon = 1.2   #3   #1.2  # Adjust based on the scale of your data
clustering_min_samples = 8  #8 # Start with a reasonable guess and adjust as needed



save_dir = '/mnt/HDD2/Channel_Sensing_Raw_Data/Experiments_Anechoic_1/Saved_Files/40MHz/4_Ant/Injector_1/Location_1/' + str(delta_t) + '/'


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


def plot_mdtrack_results(amplitude_list, toa_list, aoa_list, num_paths_plot, threshold, clustering_epsilon,
                         clustering_min_samples):
    plt.figure()
    vmin = threshold * 10
    vmax = 0
    data = []
    data_cluster = []
    for i in range(len(amplitude_list)):  # number of packets
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


        data_points = np.column_stack((toa_array[:num_paths_plot], aoa_array[:num_paths_plot]))
        data.append(data_points)
        result = np.concatenate(data, axis=0)


        data_points_cluster = np.column_stack((toa_array[:num_paths_plot]*1E9, aoa_array[:num_paths_plot]))
        data_cluster.append(data_points_cluster)
        result_cluster = np.concatenate(data_cluster, axis=0)


    # Perform DBSCAN clustering

    db = DBSCAN(eps=clustering_epsilon, min_samples=clustering_min_samples).fit(result_cluster)

    # Get cluster labels (-1 indicates noise/outliers)
    labels = db.labels_

    # Number of clusters in the data (ignoring noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Define a minimum cluster size threshold
    min_cluster_size = a  # You can adjust this value as needed

    # Initialize a dictionary to store cluster sizes
    cluster_sizes = {}

    # Calculate the size of each cluster
    for label in set(labels):
        if label != -1:
            cluster_sizes[label] = np.sum(labels == label)

    # Plot the clusters with sufficient data points
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]


    # Initialize dictionaries to store cluster averages
    cluster_x_avg = {}
    cluster_y_avg = {}
    cluster_min_x = {}
    cluster_max_x = {}
    cluster_min_y = {}
    cluster_max_y = {}

    # Initialize a dictionary to map original cluster labels to new labels
    label_mapping = {}
    new_label = 1

    for k, col in zip(unique_labels, colors):
        if k == -1 or cluster_sizes.get(k, 0) < min_cluster_size:
            # Black used for noise.
            #col = [0, 0, 0, 1]
            continue
        # elif cluster_sizes[k] < min_cluster_size:
        #     continue  # Skip clusters with fewer data points than the threshold

        class_member_mask = (labels == k)

        xy = result[class_member_mask]

        # Print the average values of the clusters'

        x_values = xy[:, 0]
        y_values = xy[:, 1]

        # Calculate the average x and y values for the cluster
        x_avg = np.mean(x_values)
        y_avg = np.mean(y_values)

        # Calculate the minimum and maximum X and Y values for the cluster
        min_x = np.min(x_values)
        max_x = np.max(x_values)
        min_y = np.min(y_values)
        max_y = np.max(y_values)

        if x_avg >= 3e-9:
            # Store the averages in the dictionaries
            cluster_x_avg[new_label] = x_avg
            cluster_y_avg[new_label] = y_avg

            cluster_min_x[new_label] = min_x
            cluster_max_x[new_label] = max_x
            cluster_min_y[new_label] = min_y
            cluster_max_y[new_label] = max_y

            # Store the mapping from original label to new label
            label_mapping[k] = new_label

            # Print the label and average values for the cluster
            print(f"Cluster {new_label}:")
            print(f"  Average X: {x_avg*3e8} m")
            print(f"  Average Y: {y_avg} degree")
            print(f"  Minimum X: {min_x*3e8} m")
            print(f"  Maximum X: {max_x*3e8} m")
            print(f"  Minimum Y: {min_y} degree")
            print(f"  Maximum Y: {max_y} degree")
            print()  # Print a blank line for separation

            new_label += 1  # Increment the new label

        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    # Set y-axis limits to -90 and 90
    plt.ylim(-90, 90)
    plt.xlim(3e-9, 30e-9)
    plt.grid()
    plt.xlabel('ToA [ns]', fontsize=15)
    plt.ylabel('AoA [deg]', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('AoA [deg]', fontsize=15)
    plt.title('Object detection Clusters', fontsize=15)
    plt.show()


plot_mdtrack_results(amplitude_list, toa_list, aoa_list, num_paths_plot, threshold, clustering_epsilon, clustering_min_samples)
