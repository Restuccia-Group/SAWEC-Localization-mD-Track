
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
import csiread

from utilityfunct_aoa_toa_doppler import build_aoa_matrix, build_toa_matrix
from utilityfunct_md_track import md_track_2d
import matplotlib
# matplotlib.use('QtCairo')


def plot_mdtrack_results(amplitude_list, toa_list, aoa_list):
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
        num_paths_plot = 5
        # print(paths_power[:num_paths_plot])
        aoa_array = paths_aoa_sort #- paths_aoa_sort[0]
        # aoa_array[aoa_array > 90] = aoa_array[aoa_array > 90] - 180
        # aoa_array[aoa_array < -90] = 180 + aoa_array[aoa_array < -90]
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
    title = 'Antennas idx ' + str(antennas_idx_considered)
    plt.title(title)
    plt.grid()
    # plt.scatter(paths_refined_toa_array[:20], paths_refined_aoa_array[:20])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exp_dir', help='Name base of the directory')
    parser.add_argument('calibration_dir', help='Name base of the directory for calibration')
    parser.add_argument('name_base', help='Name base of the simulation')
    parser.add_argument('--delta_t', help='Delta ToA for grid search in multiples of 10^-11', default=50, type=int, required=False)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    calibration_dir = args.calibration_dir
    name_base = args.name_base  # simulation

    delta_t = np.round(args.delta_t * 1e-11, 11)  # 5E-10  # 1.25e-11
    save_dir = '../results/mdTrack_unipd' + str(delta_t) + '/'
    if os.path.exists(save_dir + 'paths_list_' + name_base + '.txt'):
        print('Already processed')
        exit()

    # DATA LOADING
    signal_complete = []

    num_ant = 4

    BW = 80
    csi_file = exp_dir + 'trace3.pcap'
    csi_file_calib = calibration_dir + 'trace3.pcap'
    try:
        csi_data = csiread.Nexmon(csi_file, chip='4366c0', bw=BW)
        csi_data.read()
        signal_raw = csi_data.csi[:11000]
        signal_raw = np.fft.fftshift(signal_raw, axes=1)
        
        csi_calibration = csiread.Nexmon(csi_file_calib, chip='4366c0', bw=BW)
        csi_calibration.read()
        signal_calibration = csi_calibration.csi
        signal_calibration = np.fft.fftshift(signal_calibration, axes=1)
        
    except Exception:
        print('error in this packet, skipping...')

    # plt.figure()
    # plt.pcolor(abs(signal_calibration[:500, :]).T)
    # plt.show()
    # plt.figure()
    # plt.pcolor(np.angle(signal_calibration[:500, :]).T)
    # plt.show()
    # plt.figure()
    # plt.plot(np.unwrap(np.angle(signal_calibration[:500, :])).T)
    # plt.show()

    F_frequency = 256  # 1996 without pilot probably
    delta_f = 312.5E3

    frequency_vector_idx = np.arange(F_frequency)
    frequency_vector_hz = delta_f * (frequency_vector_idx - F_frequency / 2)

    control_subchannels = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253,
                              254, 255], dtype=int)
    
    frequency_vector_idx = np.delete(frequency_vector_idx, delete_idxs)
    # frequency_vector_idx = np.concatenate((np.arange(-1012, -514),
    #                                       np.arange(-509, -11),
    #                                       np.arange(12, 510),
    #                                       np.arange(515, 1013))) + 1024

    frequency_vector_hz = frequency_vector_hz[frequency_vector_idx]

    H_complete_valid = []
    num_time_steps_tot = signal_raw.shape[0]
    for stream in range(num_ant):

        signal_stream = signal_raw[stream:num_time_steps_tot * num_ant:num_ant, :]
        signal_stream[:, 64:] = - signal_stream[:, 64:]  # TODO needed (already checked)
        signal_stream = np.delete(signal_stream, delete_idxs, axis=1)

        signal_stream_calib = signal_calibration[stream:num_time_steps_tot * num_ant:num_ant, :]
        signal_stream_calib[:, 64:] = - signal_stream_calib[:, 64:]  # TODO needed (already checked)
        signal_stream_calib = np.delete(signal_stream_calib, delete_idxs, axis=1)

        # signal_stream_calibrated = signal_stream * np.conj(signal_stream_calib[100, :])
        signal_stream_calibrated = signal_stream / signal_stream_calib[0, :]
        
        # signal_considered = np.delete(signal_considered, delete_idxs, axis=1)
        mean_signal = np.mean(np.abs(signal_stream_calibrated), axis=1, keepdims=True)
        signal_stream_calibrated = signal_stream_calibrated / mean_signal

        delete_zeros_idxs = np.argwhere(np.sum(signal_stream_calibrated, axis=1) == 0)[:, 0]
        signal_stream_calibrated = np.delete(signal_stream_calibrated, delete_zeros_idxs, axis=0)
        
        H_complete_valid.append(signal_stream_calibrated)
        # plt.figure(); plt.stem(abs(np.fft.ifft(signal_stream_calibrated[3, :]))); plt.show();
    
    H_complete_valid = np.stack(H_complete_valid, axis=2)
    num_time_steps = H_complete_valid.shape[0]

    antennas_idx_considered = [0, 1, 2, 3]
    H_complete_valid = H_complete_valid[:, :, antennas_idx_considered]
    num_ant = H_complete_valid.shape[2]

    fc = 5775E6
    frequency_vector_hz = frequency_vector_hz + fc

    T = 1/delta_f  # OFDM symbol time
    range_considered = 3e-7
    idxs_range_considered = int(range_considered/delta_t + 1)
    t_min = -T/4
    t_max = T/4  # T/2

    num_angles = 360
    num_subc = frequency_vector_idx.shape[0]
    ToA_matrix, time_vector = build_toa_matrix(frequency_vector_hz, delta_t, t_min, t_max)
    AoA_matrix, angles_vector, cos_ant_vector = build_aoa_matrix(num_angles, num_ant)
    AoA_matrix_reshaped = np.reshape(AoA_matrix, (AoA_matrix.shape[0], -1))

    # mD-track 2D: remove offsets CFO, PDD, SFO
    paths_list = []
    paths_amplitude_list = []
    paths_toa_list = []
    paths_aoa_list = []
    optimization_times = np.zeros(num_time_steps)

    num_iteration_refinement = 10
    threshold = -2.5
    
    for time_idx in range(0, num_time_steps):
        # time_start = time.time()
        cfr_sample = H_complete_valid[time_idx, :, :]

        # plt.figure()
        # cir = np.fft.fftshift(np.fft.fft2(cfr_sample, s=(2048 * 4, 2048)), axes=(1, 0))
        # plt.pcolor(abs(cir[2048 * 2 - 200:2048 * 2 + 200, :]).T)
        # plt.show()
        # plt.figure()
        # plt.plot(abs(H_complete_valid[time_idx, :, 0]))
        # plt.show()

        # coarse estimation
        matrix_cfr_toa = np.dot(ToA_matrix, cfr_sample)
        power_matrix_cfr_toa = np.sum(np.abs(matrix_cfr_toa), 1)
        time_idx_max = np.argmax(power_matrix_cfr_toa)
        time_max = time_vector[time_idx_max]
        index_start_toa = int(max(0, time_idx_max - idxs_range_considered))
        index_end_toa = int(min(time_vector.shape[0], time_idx_max + idxs_range_considered))
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
        # plot_mdtrack_results(paths_amplitude_list[:time_idx], paths_toa_list[:time_idx], paths_aoa_list[:time_idx])

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
