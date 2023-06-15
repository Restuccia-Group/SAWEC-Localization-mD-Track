
"""
    Copyright (C) 2023 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import scipy.io as sio
import numpy as np
import time
import pickle
import math as mt
import matplotlib.pyplot as plt
import os

from utilityfunct_aoa_toa_doppler import build_aoa_matrix, build_toa_matrix, build_dop_matrix
from utilityfunct_md_track import md_track_3d


def plot_combined(paths_refined_amplitude_array, paths_refined_toa_array, paths_refined_aoa_array,
                  path_loss_sorted_sim, times_sorted_sim, azimuth_sorted_sim_2):
    vmin = -40
    vmax = 0
    plt.figure(figsize=(5, 4))

    # plot ground truth
    if path_loss_sorted_sim is not None:
        paths_power = - path_loss_sorted_sim + path_loss_sorted_sim[:, 0]  # dB
        paths_power = paths_power[0, :]
    else:
        paths_power = np.ones_like(azimuth_sorted_sim_2)
    toa_array = times_sorted_sim
    plt.scatter(toa_array * 1E9, azimuth_sorted_sim_2,
                c=paths_power,
                marker='o', cmap='Blues', s=20,
                vmin=vmin, vmax=vmax, label='ground')

    cbar = plt.colorbar()

    # plot sim
    paths_power = np.power(np.abs(paths_refined_amplitude_array), 2)
    paths_power = 10 * np.log10(paths_power / np.amax(np.nan_to_num(paths_power)))  # dB
    toa_array = paths_refined_toa_array  # - paths_refined_toa_array[0]
    plt.scatter(toa_array * 1E9, paths_refined_aoa_array,
                c=paths_power,
                marker='x', cmap='Reds', s=20,
                vmin=vmin, vmax=vmax, label='mdTrack')

    cbar.ax.set_ylabel('power [dB]', rotation=90)
    plt.xlabel('ToA [ns]')
    plt.ylabel('AoA [deg]')
    plt.ylim([-90, 90])
    plt.yticks(np.arange(-90, 91, 20))
    plt.grid()
    plt.legend(prop={'size': 10})
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    parser.add_argument('exp_dir', help='Name base of the directory')
    parser.add_argument('name_base', help='Name base of the simulation')
    parser.add_argument('--delta_t', help='Delta ToA for grid search in multiples of 10^-11', default=50, type=int, required=False)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    name_base = args.name_base  # simulation

    delta_t = np.round(args.delta_t * 1e-11, 11)  # 5E-10  # 1.25e-11
    save_dir = '../results/mdTrack' + str(delta_t) + '/'
    if os.path.exists(save_dir + 'paths_list_' + name_base + '.txt'):
        print('Already processed')
        exit()

    num_ant = args.nss * args.ncore

    # DATA LOADING
    # TODO now each file is [number_of_frames X number_of_subcarriers]
    # TODO create a matrix [number_of_frames X number_of_subcarriers X (number_of_azimuth_antennas + number_of_elevation_antennas)]
    # TODO create a matrix [number_of_doppler_elem X number_of_subcarriers X (number_of_azimuth_antennas + number_of_elevation_antennas)]
    signal_complete = []

    A05, A04, A03, A02, A01 = 0, 0, 0, 0, 0

    A = ['A05', 'A04', 'A03', 'A02', 'A01']

    for file_name in A:
        csi_file = exp_dir + file_name + '.npy'
        globals()[file_name] = np.load(csi_file)
        globals()[file_name] = globals()[file_name][:11000]

    AA = [A05, A04, A03, A02, A01]

    signal_complete = np.stack(AA, axis=2)

    num_time_steps = signal_complete.shape[0]

    Tc = 5e-3  # sampling time

    F_frequency = 2025
    delta_f = 78.5E3  # sub-channel spacing

    frequency_vector_idx = np.arange(F_frequency)
    frequency_vector_hz = delta_f * (frequency_vector_idx - F_frequency / 2)
    # control_subcarriers = [0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255]
    # pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231]
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253,
                              254, 255], dtype=int)
    # frequency_vector_idx = np.delete(frequency_vector_idx, delete_idxs, axis=0)
    # frequency_vector_hz = np.delete(frequency_vector_hz, delete_idxs, axis=0)
    fc = 5180E6
    frequency_vector_hz = frequency_vector_hz + fc

    H_complete_valid = signal_complete

    T = 1/delta_f  # OFDM symbol time
    range_considered = 1e-8
    idxs_range_considered = int(range_considered/delta_t + 1)
    t_min = 0
    t_max = T / 3  # T/2

    num_angles = 360
    num_paths = 100
    num_subc = frequency_vector_idx.shape[0]
    ToA_matrix, time_vector = build_toa_matrix(frequency_vector_hz, delta_t, t_min, t_max)
    AoA_matrix, angles_vector, cos_ant_vector = build_aoa_matrix(num_angles, num_ant)
    AoA_matrix_reshaped = np.reshape(AoA_matrix, (AoA_matrix.shape[0], -1))

    num_pkts = 50
    step = 1
    Dop_matrix, doppler_vector = build_dop_matrix(num_pkts, Tc, step)
    num_freq = doppler_vector.shape[0]

    # mD-track 2D: remove offsets CFO, PDD, SFO
    paths_list = []
    paths_amplitude_list = []
    paths_toa_list = []
    paths_aoa_list = []
    paths_dop_list = []
    optimization_times = np.zeros(num_time_steps)

    num_iteration_refinement = 10
    threshold = -2.5

    for time_idx in range(0, num_time_steps, 1):
        # time_start = time.time()
        cfr_sample = H_complete_valid[time_idx:time_idx + num_pkts, :, :]

        # coarse estimation
        matrix_cfr_toa = np.dot(ToA_matrix, cfr_sample)
        power_matrix_cfr_toa = np.sum(np.abs(matrix_cfr_toa), 1)
        time_idx_max = np.argmax(power_matrix_cfr_toa)
        time_max = time_vector[time_idx_max]
        index_start_toa = int(max(0, time_idx_max - idxs_range_considered))
        index_end_toa = int(min(time_vector.shape[0], time_idx_max + idxs_range_considered))
        ToA_matrix_considered = ToA_matrix[index_start_toa:index_end_toa, :]
        time_vector_considered = time_vector[index_start_toa:index_end_toa]
        num_times = time_vector_considered.shape[0]

        #####
        # MULTI-PATH PARAMETERS ESTIMATION
        start = time.time()

        paths, paths_refined_amplitude, paths_refined_dop_idx, paths_refined_toa_idx, paths_refined_aoa_idx = \
            md_track_3d(cfr_sample, AoA_matrix, ToA_matrix_considered, Dop_matrix, num_ant, num_subc, num_angles,
                        num_pkts, num_times, num_freq, num_iteration_refinement, threshold)

        end = time.time()
        optimization_times[time_idx] = end-start

        paths_refined_aoa = angles_vector[paths_refined_aoa_idx] * 180 / mt.pi
        paths_refined_toa = time_vector_considered[paths_refined_toa_idx]
        paths_refined_dop = doppler_vector[paths_refined_dop_idx]
        paths_refined_amplitude_array = np.asarray(paths_refined_amplitude)
        paths_refined_aoa_array = np.asarray(paths_refined_aoa)
        paths_refined_toa_array = np.asarray(paths_refined_toa)
        paths_refined_dop_array = np.asarray(paths_refined_dop)

        paths_list.append(paths)
        paths_amplitude_list.append(paths_refined_amplitude_array)
        paths_aoa_list.append(paths_refined_aoa_array)
        paths_toa_list.append(paths_refined_toa_array)
        paths_dop_list.append(paths_refined_dop_array)
        #####

    # Saving results
    save_name = save_dir + 'opr_sim_' + name_base + '.txt'  # + '.npz'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_name, "wb") as fp:  # Pickling
        pickle.dump(optimization_times, fp)

    name_file = save_dir + 'paths_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_list, fp)
    name_file = save_dir + 'paths_amplitude_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_amplitude_list, fp)
    name_file = save_dir + 'paths_aoa_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_aoa_list, fp)
    name_file = save_dir + 'paths_toa_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_toa_list, fp)
    name_file = save_dir + 'paths_dop_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_dop_list, fp)

    # plot_combined(paths_refined_amplitude_array, paths_refined_toa_array, paths_refined_aoa_array,
    #               path_loss_sorted_sim, times_sorted_sim, azimuth_sorted_sim_2)
