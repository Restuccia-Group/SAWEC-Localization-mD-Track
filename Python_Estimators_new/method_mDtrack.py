
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

from utilityfunct_aoa_toa_doppler import build_aoa_matrix, build_toa_matrix
from utilityfunct_md_track import md_track_2d
import matplotlib
# matplotlib.use('QtCairo')

def plot_mdtrack_results(amplitude_list, toa_list, aoa_list, num_paths_plot):
    plt.figure()
    vmin = threshold*10
    vmax = 0
    end_plot = len(amplitude_list)
    for i in range(0, end_plot):  # number of packets
        sort_idx = np.flip(np.argsort(abs(amplitude_list[i])))
        paths_amplitude_sort = amplitude_list[i][sort_idx]
        paths_power = np.power(np.abs(paths_amplitude_sort), 2)
        paths_power = 10 * np.log10(paths_power / np.amax(np.nan_to_num(paths_power)))  # dB
        paths_toa_sort = toa_list[i][sort_idx]
        paths_aoa_sort = aoa_list[i][sort_idx]
        # print(paths_power[:num_paths_plot])
        #aoa_array = paths_aoa_sort #- paths_aoa_sort[0]
        # aoa_array[aoa_array > 90] = aoa_array[aoa_array > 90] - 180
        # aoa_array[aoa_array < -90] = 180 + aoa_array[aoa_array < -90]



        #print(paths_aoa_sort)
        aoa_array = paths_aoa_sort - paths_aoa_sort[0]
        #print(paths_aoa_sort[0])
        #print(aoa_array)

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
    # plt.xlim([-10, 20])  # range_considered + 100 * delta_t])
    # plt.ylim([-90, 90])
    plt.grid()
    # plt.scatter(paths_refined_toa_array[:20], paths_refined_aoa_array[:20])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nics', help='Name of the files, comma separated')
    parser.add_argument('fc', help='Central frequency in MHz', type=int)
    parser.add_argument('BW', help='Bandwidth in MHz', type=int)
    parser.add_argument('exp_dir', help='Name base of the directory')
    parser.add_argument('calibration_dir', help='Name base of the directory for calibration')
    parser.add_argument('name_base', help='Name base of the simulation')
    parser.add_argument('--delta_t', help='Delta ToA for grid search in multiples of 10^-11', default=50, type=int, required=False)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    calibration_dir = args.calibration_dir
    name_base = args.name_base  # simulation

    BW = args.BW * 1e6
    if BW == 20E6:   # 802.11n
        F_frequency = 64  # 1996 without pilot probably
        delta_f = 312.5E3
        control_subchannels = np.asarray([0, 1, 2, 3, 4, 5, 59, 60, 61, 62, 63], dtype=int)
        junk_subchannel = [25]

    elif BW == 160E6:  # 802.11ax
        F_frequency = 2048  # 1996 without pilot probably
        delta_f = 78.125E3
        control_subchannels = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                          2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047], dtype=int)
        # pilot_subchannels = [-468, -400, -334, -266, -226, -158, -92, -24, 24, 92, 158, 226, 266, 334, 400, 468]
        junk_subchannel = np.arange(1996, 2025)  # PicoScenes return zero here

    elif BW == 80E6:  # 802.11ax
        F_frequency = 1024
        delta_f = 78.125E3
        control_subchannels = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                          1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023], dtype=int)
        junk_subchannel = []  # PicoScenes return zero here

    elif BW == 40E6:  # 802.11ax
        F_frequency = 512
        delta_f = 78.125E3
        control_subchannels = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                          501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511], dtype=int)
        junk_subchannel = []  # PicoScenes return zero here

    delta_t = np.round(args.delta_t * 1e-11, 11)  # 5E-10  # 1.25e-11
    save_dir = '../results/mdTrack' + str(delta_t) + '/'
    if os.path.exists(save_dir + 'paths_list_' + name_base + '.txt'):
        print('Already processed')
        exit()

    # DATA LOADING
    # TODO now each file is [number_of_frames X number_of_subcarriers]
    # TODO create a matrix [number_of_frames X number_of_subcarriers X (number_of_azimuth_antennas + number_of_elevation_antennas)]
    signal_complete = []

    nics_list = []  # 'A05', 'A04', 'A03', 'A02', 'A01'
    nics = args.nics
    files_name = []
    for nic in nics.split(','):
        nics_list.append(nic)
    num_ant = len(nics_list)

    singal_nics = []
    for file_name in nics_list:
        csi_file = exp_dir + file_name + '.npy'
        csi_file_calib = calibration_dir + file_name + '.npy'
        # globals()[file_name] = np.load(csi_file)
        # globals()[file_name] = globals()[file_name][:11000]
        signal_raw = np.load(csi_file)[:11000]
        signal_calibration = np.load(csi_file_calib)
        # signal_calibrated = signal_raw * np.conj(signal_calibration[100])
        signal_calibrated = signal_raw / signal_calibration[0, :]
        singal_nics.append(signal_calibrated)

        # plt.figure()
        # plt.pcolor(abs(signal_calibration[:500, :]).T)
        # plt.show()
        # plt.figure()
        # plt.pcolor(np.angle(signal_calibration[:500, :]).T)
        # plt.show()
        # plt.figure()
        # plt.plot(np.unwrap(np.angle(signal_calibration[:500, :])).T)
        # plt.show()
        # plt.figure()
        # plt.plot(np.abs(signal_calibration[100, :]).T)
        # plt.show()

    signal_complete = np.stack(singal_nics, axis=2)

    num_time_steps = signal_complete.shape[0]

    frequency_vector_idx = np.arange(F_frequency)
    frequency_vector_hz = delta_f * (frequency_vector_idx - F_frequency / 2)

    frequency_vector_idx = np.delete(frequency_vector_idx, control_subchannels)
    frequency_vector_hz = np.delete(frequency_vector_hz, control_subchannels)  # frequency_vector_hz[frequency_vector_idx]

    frequency_vector_idx = np.delete(frequency_vector_idx, junk_subchannel)
    frequency_vector_hz = np.delete(frequency_vector_hz, junk_subchannel)
    H_complete_valid = np.delete(signal_complete, junk_subchannel, axis=1)  # packets, subchannels, angles

    # plt.figure()
    # plt.plot(np.abs(H_complete_valid[20:30, :, 0]).T)
    # plt.ylim([-0.5, 2.5])
    # plt.show()

    fc = args.fc * 1e6  # B: 5570E6  # A: 5250E6 # carrier frequency in MHz
    frequency_vector_hz = frequency_vector_hz + fc
    T = 1/delta_f  #

    range_refined_up = 4E-8  # 2.5E-7
    idxs_range_ref_up = int(range_refined_up / delta_t + 1)
    range_refined_down = 1E-8  # 2.5E-7
    idxs_range_ref_down = int(range_refined_down / delta_t + 1)
    t_min = -T/4
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

        # plt.figure()
        # cir = np.fft.fftshift(np.fft.fft2(cfr_sample, s=(2048 * 4, 2048)), axes=(1, 0))
        # plt.pcolor(abs(cir[2048 * 2 - 200:2048 * 2 + 200, :]).T)
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

        a = 1
        # PLOT FOR DEBUG
        # num_paths_plot = 5
        # start_plot = 0
        # end_plot = 100 #len(paths_amplitude_list)
        # plot_mdtrack_results(paths_amplitude_list[start_plot:end_plot], paths_toa_list[start_plot:end_plot], paths_aoa_list[start_plot:end_plot], num_paths_plot)

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

    # plt.figure()
    # plt.stem(abs(paths_refined_amplitude_array))
    # plt.show()
    #
    # plt.figure()
    # plt.stem(abs(power_matrix_cfr_toa[:1000]))
    # plt.show()
