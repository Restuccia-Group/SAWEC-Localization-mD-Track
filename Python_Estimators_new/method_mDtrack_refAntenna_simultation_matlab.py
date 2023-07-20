
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
import scipy.io

from utilityfunct_aoa_toa_doppler import build_aoa_matrix, build_toa_matrix
from utilityfunct_md_track import md_track_2d
import matplotlib
# matplotlib.use('QtCairo')


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
    parser.add_argument('fc', help='Central frequency in MHz', type=int)
    parser.add_argument('BW', help='Bandwidth in MHz', type=int)
    parser.add_argument('exp_dir', help='Name base of the directory')
    parser.add_argument('max_order_reflection', help='Maximum number of reflectors')
    parser.add_argument('--delta_t', help='Delta ToA for grid search in multiples of 10^-11',
                        default=50, type=int, required=False)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    file_name = args.file_name  # 'chanEstArray_1x8.mat'

    BW = args.BW * 1e6
    if BW == 20E6:
        F_frequency = 64  # 1996 without pilot probably
        delta_f = 312.5E3
        control_subchannels = np.asarray([0, 1, 2, 3, 4, 5, 10, 24, 31, 38, 52, 59, 60, 61, 62, 63], dtype=int)

    elif BW == 160E6:
        F_frequency = 512  # 1996 without pilot probably
        delta_f = 312.5E3
        control_subchannels = np.asarray([0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253, 254,
                              255, 256, 257, 258, 259, 260, 261, 281, 309, 345, 373, 383, 384, 385, 395, 423, 459, 487,
                              507, 508, 509, 510, 511], dtype=int)

    name_base = 'simulation_office'

    # delta_t for time granularity
    delta_t = np.round(args.delta_t * 1e-11, 11)  # 5E-10  # 1.25e-11
    save_dir = '../results/mdTrack' + str(delta_t) + '/'
    if os.path.exists(save_dir + 'paths_list_' + name_base + '.txt'):
        print('Already processed')
        exit()

    # DATA LOADING
    folder_name = args.exp_dir  # '../files_matlab/20mhz/'

    max_order_reflection = args.max_order_reflection  # 1
    name_file = folder_name + 'CFR_' + name_base + '_160_maxref' + str(max_order_reflection) + '.mat'
    csi_buff = sio.loadmat(name_file)
    signal_cfr = (csi_buff['CFR'])

    antennas_idx_considered = [0, 1, 2, 3, 4, 5, 6, 7]
    signal_complete = signal_cfr[:, :, antennas_idx_considered]
    signal_complete = np.moveaxis(signal_complete, [0, 1], [1, 0])
    num_time_steps = signal_complete.shape[0]
    num_ant = signal_complete.shape[2]

    name_file = folder_name + 'delay_' + name_base + '_160_maxref' + str(max_order_reflection) + '.mat'
    csi_buff = sio.loadmat(name_file)
    delays_sim = (csi_buff['propagation_delays'])

    name_file = folder_name + 'aoa_' + name_base + '_160_maxref' + str(max_order_reflection) + '.mat'
    csi_buff = sio.loadmat(name_file)
    aoa_sim = (csi_buff['propagation_aoa'])

    name_file = folder_name + 'path_loss_' + name_base + '_160_maxref' + str(max_order_reflection) + '.mat'
    csi_buff = sio.loadmat(name_file)
    path_loss_sim = (csi_buff['propagation_path_loss'])

    # FIGURES FOR DEBUG
    # plt.figure()
    # plt.pcolor(np.abs(signal_complete[100:1200, :, 2]).T, vmax=3)
    # plt.colorbar()
    # plt.show()
    # plt.figure()
    # plt.pcolor(np.abs(singal_nics[2][280:300, :]).T, vmax=3)
    # plt.colorbar()
    # plt.figure()
    # plt.pcolor(np.abs(singal_nics_ref[2][280:300, :]).T, vmax=3)
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.plot(np.abs(singal_nics[0][100, :]).T)
    # plt.show()
    # plt.figure()
    # plt.plot(np.abs(singal_nics[1][100, :]).T)
    # plt.plot(np.abs(singal_nics[3][100, :]).T)
    # plt.show()
    # plt.figure()
    # plt.plot(np.abs(signal_nic_calibrated[3][100, :]).T)
    # plt.plot(np.abs(signal_nic_calibrated[1][100, :]).T)
    # plt.show()
    # plt.figure()
    # plt.plot(np.abs(singal_nics[2][100, :]).T)
    # plt.show()
    # plt.figure()
    # plt.plot(np.abs(offset_wireless[100:105, :]).T)
    # plt.show()

    # WIRELESS PARAMETERS
    frequency_vector_idx = np.arange(F_frequency)
    frequency_vector_hz = delta_f * (frequency_vector_idx - F_frequency / 2)

    frequency_vector_idx = np.delete(frequency_vector_idx, control_subchannels)
    frequency_vector_hz = np.delete(frequency_vector_hz, control_subchannels)

    delete_idxs = [25]  # center frequency
    frequency_vector_idx = np.delete(frequency_vector_idx, delete_idxs)
    frequency_vector_hz = np.delete(frequency_vector_hz, delete_idxs)

    H_complete_valid = signal_complete # packets, subchannels, angles

    # plt.figure()
    # plt.plot(np.abs(H_complete_valid[20:30, :, 0]).T)
    # plt.ylim([-0.5, 2.5])
    # plt.show()

    fc = args.fc * 1e6
    frequency_vector_hz = frequency_vector_hz + fc
    T = 1/delta_f

    # MDTRACK PARAMETERS
    range_refined_up = 4E-8  # 2.5E-7
    idxs_range_ref_up = int(range_refined_up / delta_t + 1)
    range_refined_down = 1E-8  # 2.5E-7
    idxs_range_ref_down = int(range_refined_down / delta_t + 1)
    t_min =  -T/4
    t_max = T/4  # T/2

    num_angles = 360
    num_subc = frequency_vector_idx.shape[0]
    ToA_matrix, time_vector = build_toa_matrix(frequency_vector_hz, delta_t, t_min, t_max)
    AoA_matrix, angles_vector, sin_ant_vector = build_aoa_matrix(num_angles, num_ant)

    # mD-track 2D: remove offsets CFO, PDD, SFO
    paths_list = []
    paths_amplitude_list = []
    paths_toa_list = []
    paths_aoa_list = []
    optimization_times = np.zeros(num_time_steps)

    num_iteration_refinement = 10
    threshold = -2.5  # - 25 dB

    for time_idx in range(0, num_time_steps):
        # time_start = time.time()
        cfr_sample = H_complete_valid[time_idx, :, :]
        cfr_sample = np.nan_to_num(cfr_sample, posinf=0, neginf=0)

        # FGURES FOR DEBUG
        # plt.figure()
        # cir = np.fft.fftshift(np.fft.fft2(cfr_sample, s=(2048 * 4, 2048)), axes=(1, 0))
        # plt.pcolor(abs(cir[2048 * 2 - 200:2048 * 2 + 200, :]).T)
        # plt.show()
        # plt.figure()
        # cir = np.fft.fftshift(np.fft.fft(cfr_sample[:, 0], 200))
        # plt.stem(abs(cir[60:200]).T)
        # plt.show()

        # plt.figure()
        # plt.plot(np.abs(offset_wireless[100:105, :]).T)
        # plt.show()
        # plt.figure()
        # plt.plot(np.abs(offset_wireless[100:105, :]).T)
        # plt.show()

        # coarse estimation
        matrix_cfr_toa = np.dot(ToA_matrix, cfr_sample)
        power_matrix_cfr_toa = np.sum(np.abs(matrix_cfr_toa), 1)
        time_idx_max = np.argmax(power_matrix_cfr_toa)
        time_max = time_vector[time_idx_max]
        index_start_toa = int(max(0, time_idx_max - idxs_range_ref_down))
        index_end_toa = int(min(time_vector.shape[0], time_idx_max + idxs_range_ref_up))
        ToA_matrix_considered = ToA_matrix[index_start_toa:index_end_toa, :]
        time_vector_considered = time_vector[index_start_toa:index_end_toa]

        #####
        # MULTI-PATH PARAMETERS ESTIMATION
        start = time.time()

        paths, paths_refined_amplitude, paths_refined_toa_idx, paths_refined_aoa_idx = md_track_2d(
            cfr_sample, AoA_matrix, ToA_matrix_considered, num_ant, num_subc, num_angles, num_iteration_refinement,
            threshold)
        end = time.time()
        optimization_times[time_idx] = end-start

        paths_refined_aoa = angles_vector[paths_refined_aoa_idx] * 180 / mt.pi
        paths_refined_toa = time_vector_considered[paths_refined_toa_idx]
        paths_refined_amplitude_array = np.asarray(paths_refined_amplitude)
        paths_refined_aoa_array = np.asarray(paths_refined_aoa)
        paths_refined_toa_array = np.asarray(paths_refined_toa)

        paths_list.append(paths)
        paths_amplitude_list.append(paths_refined_amplitude_array)
        paths_aoa_list.append(paths_refined_aoa_array)
        paths_toa_list.append(paths_refined_toa_array)
        #####

        #####
        # GROUND TRUTH SIMULATION
        sorted_idx_sim = np.argsort(abs(path_loss_sim[0, time_idx]))[0, :]

        elevation_sorted_sim = (aoa_sim[0, time_idx][1, sorted_idx_sim])
        azimuth_sorted_sim = (aoa_sim[0, time_idx][0, sorted_idx_sim])

        azimuth_sorted_sim_2 = np.arcsin(np.sin(azimuth_sorted_sim / 180 * mt.pi)
                                         * np.cos(elevation_sorted_sim / 180 * mt.pi)) * 180 / mt.pi

        az_positive = azimuth_sorted_sim_2 > 0
        az_negative = azimuth_sorted_sim_2 < 0
        azimuth_sorted_sim_2[az_positive] -= 180
        azimuth_sorted_sim_2[az_negative] += 180

        swap_idx_pos = azimuth_sorted_sim_2 > 90
        swap_idx_neg = azimuth_sorted_sim_2 < -90
        azimuth_sorted_sim_2[swap_idx_pos] = 180 - azimuth_sorted_sim_2[swap_idx_pos]
        azimuth_sorted_sim_2[swap_idx_neg] = - 180 - azimuth_sorted_sim_2[swap_idx_neg]

        times_sorted_sim = delays_sim[0, time_idx][:, sorted_idx_sim]
        path_loss_sorted_sim = path_loss_sim[0, time_idx][:, sorted_idx_sim]
        #####

        a = 1
        # PLOT FOR DEBUG
        # plot_combined(paths_refined_amplitude_array, paths_refined_toa_array - paths_refined_toa_array[0],
        #               paths_refined_aoa_array, path_loss_sorted_sim, times_sorted_sim - times_sorted_sim[:, 0],
        #               azimuth_sorted_sim_2)

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
