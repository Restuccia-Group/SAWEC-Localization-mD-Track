
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

from multiprocessing import Process, Value, Pool, Manager, Queue
import argparse
import numpy as np
import os
import csiread
from sharp import optimization_utility, data_utility
import math as mt
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from scipy.signal.windows import hann
import tensorflow as tf
import matplotlib

matplotlib.use('QtCairo')  # works with debug and run!

import matplotlib.pyplot as plt
import cmath as cmt
import time
from os import getpid
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
sg.set_options(font=('Arial'))

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.sans-serif'] = ['Times']
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['text.usetex'] = 'true'
#matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
matplotlib.rcParams['axes.linewidth'] = 0.5 #default 0.
matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)
matplotlib.rcParams['toolbar'] = 'None'


def plots_antennas(q, i):
    # PLOTS
    plt.ion()
    fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(6.7, 4.8))
    fig_canvas = fig.canvas  # fig_canvas = FigureCanvasTkAgg(fig)

    # To place the figures
    num_columns = 4
    start_x = 560
    start_y = 115
    x = start_x + 730 * int(mt.fmod(i, num_columns))
    y = start_y + 480 * int(i//num_columns)
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        fig_canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        # fig_canvas.manager.window.overrideredirect(True)
    elif backend == 'WXAgg':
        fig_canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        fig_canvas.manager.window.move(x, y)

    # print('I am number %d in process %d - plots' % (i, getpid()))
    # PLOT
    ax0.set_ylim(0, 3)
    ax0.set_xlim(0, F_frequency_retained)
    ax0.set_xticks(np.arange(0, F_frequency_retained + 1, 22))
    ax0.grid(axis='both')
    ax0.set_ylabel(r'amplitude')
    name_plot = r'\textbf{antenna idx: %d}' % i
    ax0.set_title(name_plot)

    ax1.set_ylim(-100, 100)
    ax1.set_xlim(0, F_frequency_retained)
    ax1.set_xticks(np.arange(0, F_frequency_retained + 1, 22))
    ax1.grid(axis='both')
    ax1.set_xlabel(r'OFDM sub-channel index')
    ax1.set_ylabel(r'phase')

    x1 = np.arange(0, F_frequency_retained)
    linea, = ax0.plot(x1, [np.nan] * F_frequency_retained, linewidth=2.0, color='#0C2C52')
    lineb, = ax1.plot(x1, [np.nan] * F_frequency_retained, linewidth=2.0, color='#0C2C52')
    fig_canvas.draw()  # draw and show it
    plt.show(block=False)

    while True:
        signal_considered = q.get() + 0
        for time_i in range(0, signal_considered.shape[1], plot_int):
            linea.set_ydata(np.abs(signal_considered[:, time_i]))
            lineb.set_ydata(np.unwrap(np.angle(signal_considered[:, time_i]), axis=0))
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            ax0.relim()
            ax0.autoscale_view(tight=True, scalex=True, scaley=False)
            # ax1.relim()
            # ax1.autoscale_view(tight=True, scalex=True, scaley=False)
            fig_canvas.draw()
            plt.pause(1e-30)


def plots_doppler_antennas(q, i):
    # PLOTS
    plt.ion()
    fig, ax0 = plt.subplots(1, 1, figsize=(6.7, 3))
    fig_canvas = fig.canvas  # fig_canvas = FigureCanvasTkAgg(fig)

    # To place the figures
    num_columns = 4
    start_x = 560
    start_y = 115 + 480 + 50
    x = start_x + 730 * int(mt.fmod(i, num_columns))
    y = start_y + 480 * int(i // num_columns)
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        fig_canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        # fig_canvas.manager.window.overrideredirect(True)
    elif backend == 'WXAgg':
        fig_canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        fig_canvas.manager.window.move(x, y)

    # print('I am number %d in process %d - plots' % (i, getpid()))
    # PLOT
    step = 20
    step_x = 6
    length_v = mt.floor(doppler_dim / 2)
    factor_v = step * (mt.floor(length_v / step))
    ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
    ticks_x = np.arange(0, window_length + 1, int(window_length / step_x))

    ax0.set_yticks(ticks_y + 0.5)
    ax0.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
    ax0.set_xticks(ticks_x)
    ax0.set_xticklabels(np.round(ticks_x * stride_length * Tc, 2))

    ax0.grid(axis='both')
    ax0.set_ylabel(r'velocity [m/s]')
    ax0.set_xlabel(r'time [s]')
    name_plot = r'\textbf{antenna idx: %d}' % i
    ax0.set_title(name_plot)

    act_map = np.zeros((doppler_dim, window_length))
    pcol1 = ax0.imshow(act_map, cmap='viridis')
    # pcol1.set_edgecolor('face')
    fig_canvas.draw()  # draw and show it
    plt.show(block=False)

    while True:
        act_map_new = (q.get() + 0).T
        end_dopp = q.get()
        act_map[:, :window_length-end_dopp] = act_map[:, end_dopp:]
        act_map[:, window_length-end_dopp:] = np.log(act_map_new)
        ax0.imshow(act_map, cmap='viridis')
        ax0.relim()
        ax0.autoscale()
        # ax1.autoscale_view(tight=True, scalex=True, scaley=False)
        fig_canvas.draw()
        plt.pause(1e-30)


def antenna_processing(signal_considered, stream, csi_doppler_matrix, queue, queue_doppler, end_s):

    signal_considered[:, 64:] = - signal_considered[:, 64:]  # TODO needed (already checked)

    signal_considered = np.delete(signal_considered, delete_idxs, axis=1)
    mean_signal = np.mean(np.abs(signal_considered), axis=1, keepdims=True)
    signal_considered = signal_considered / mean_signal
    signal_considered = signal_considered.T

    delete_zeros_idxs = np.argwhere(np.sum(signal_considered, axis=0) == 0)[:, 0]
    signal_considered = np.delete(signal_considered, delete_zeros_idxs, axis=1)

    # plot
    queue.put(signal_considered, stream)

    """
        CSI_phase_sanitization_H_estimation_mD-Track
    """
    # print('I am number %d in process %d - started' % (stream, getpid()))
    # signal_considered = signal_complete[:, :, stream]
    Tr_matrix = np.zeros((F_frequency_retained, end_s), dtype=complex)

    for time_step in range(0, end_s, path_est_int):
        signal_time = signal_considered[:, time_step]

        matrix_cfr_toa = np.dot(ToA_matrix, signal_time)
        power_matrix_cfr_toa = np.abs(matrix_cfr_toa)
        time_idx_max = np.argmax(power_matrix_cfr_toa)  # time_max = time_vector[time_idx_max]
        index_start_toa = int(max(0, time_idx_max - idxs_range_considered))
        index_end_toa = int(min(time_vector.shape[0], time_idx_max + idxs_range_considered))
        ToA_matrix_considered = ToA_matrix[index_start_toa:index_end_toa, :]
        time_vector_considered = time_vector[index_start_toa:index_end_toa]

        paths, paths_refined_amplitude, paths_refined_toa_idx, num_estimated_paths, cfr_sample_residual = \
            optimization_utility.md_track_2d(signal_time, ToA_matrix_considered, F_frequency,
                                             num_iteration_refinement, threshold)

        paths_refined_toa = time_vector_considered[paths_refined_toa_idx]

        reconstruct_paths = np.zeros((F_frequency_retained, num_estimated_paths), dtype=complex)
        for col in range(num_estimated_paths):
            time_path = paths_refined_toa[col]
            apl_path = paths_refined_amplitude[col]
            for row in range(F_frequency_retained):
                freq_n = frequency_vector_complete_hz[row]
                reconstruct_paths[row, col] = apl_path * cmt.exp(-1j * 2 * cmt.pi * freq_n * time_path)

        signal_time_interval = signal_considered[:, time_step:min(time_step+path_est_int, end_s)]
        reconstruct_signal_clean = np.multiply(signal_time_interval,
                                               np.expand_dims(np.conj(reconstruct_paths[:, 0]), -1))
        Tr_matrix[:, time_step:min(time_step+path_est_int, end_s)] = reconstruct_signal_clean

    """ 
        CSI_phase_sanitization_signal_reconstruction
    """
    csi_matrix_processed = np.zeros((end_s, F_frequency_retained, 2))

    # AMPLITUDE
    csi_matrix_processed[:, :, 0] = np.abs(Tr_matrix).T

    # PHASE
    phase_before = np.unwrap(np.angle(Tr_matrix), axis=0)
    phase_err_tot = np.diff(phase_before, axis=1)
    ones_vector = np.ones((2, phase_before.shape[0]))
    ones_vector[1, :] = np.arange(0, phase_before.shape[0])
    for tidx in range(1, phase_before.shape[1]):
        stop = False
        idx_prec = -1
        while not stop:
            phase_err = phase_before[:, tidx] - phase_before[:, tidx - 1]
            diff_phase_err = np.diff(phase_err)
            idxs_invert_up = np.argwhere(diff_phase_err > 0.9 * mt.pi)[:, 0]
            idxs_invert_down = np.argwhere(diff_phase_err < -0.9 * mt.pi)[:, 0]
            if idxs_invert_up.shape[0] > 0:
                idx_act = idxs_invert_up[0]
                if idx_act == idx_prec:  # to avoid a continuous jump
                    stop = True
                else:
                    phase_before[idx_act + 1:, tidx] = phase_before[idx_act + 1:, tidx] \
                                                       - 2 * mt.pi
                    idx_prec = idx_act
            elif idxs_invert_down.shape[0] > 0:
                idx_act = idxs_invert_down[0]
                if idx_act == idx_prec:
                    stop = True
                else:
                    phase_before[idx_act + 1:, tidx] = phase_before[idx_act + 1:, tidx] \
                                                       + 2 * mt.pi
                    idx_prec = idx_act
            else:
                stop = True
    for tidx in range(1, Tr_matrix.shape[1] - 1):
        val_prec = phase_before[:, tidx - 1:tidx]
        val_act = phase_before[:, tidx:tidx + 1]
        error = val_act - val_prec
        temp2 = np.linalg.lstsq(ones_vector.T, error)[0]
        phase_before[:, tidx] = phase_before[:, tidx] - (np.dot(ones_vector.T, temp2)).T

    csi_matrix_processed[:, :, 1] = phase_before.T

    # print('I am number %d in process %d - step 2' % (stream, getpid()))

    """
        CSI_doppler_computation
    """
    csi_matrix_processed[:, :, 0] = csi_matrix_processed[:, :, 0] / np.mean(csi_matrix_processed[:, :, 0],
                                                                            axis=1, keepdims=True)
    csi_matrix_complete = csi_matrix_processed[:, :, 0] * np.exp(1j * csi_matrix_processed[:, :, 1])

    csi_d_profile_list = []
    for i in range(0, end_s - num_symbols, sliding):
        csi_matrix_cut = csi_matrix_complete[i:i + num_symbols, :]
        csi_matrix_cut = np.nan_to_num(csi_matrix_cut)

        hann_window = np.expand_dims(hann(num_symbols), axis=-1)
        csi_matrix_wind = np.multiply(csi_matrix_cut, hann_window)
        csi_doppler_prof = fft(csi_matrix_wind, n=doppler_dim, axis=0)
        csi_doppler_prof = fftshift(csi_doppler_prof, axes=0)

        csi_d_map = np.abs(csi_doppler_prof * np.conj(csi_doppler_prof))
        csi_d_map = np.sum(csi_d_map, axis=1)
        csi_d_profile_list.append(csi_d_map)
    csi_d_profile_array = np.asarray(csi_d_profile_list)
    csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
    csi_d_profile_array = csi_d_profile_array / csi_d_profile_array_max
    csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)
    # plt.pcolormesh(csi_d_profile_array.T, cmap='viridis', linewidth=0, rasterized=True) plt.show()
    stft_sum_1_log = csi_d_profile_array - np.mean(csi_d_profile_array, axis=0, keepdims=True)
    # plt.pcolormesh(stft_sum_1_log.T, cmap='viridis', linewidth=0, rasterized=True) plt.show()

    csi_doppler_matrix.append(stft_sum_1_log)

    queue_doppler.put(csi_d_profile_array)
    queue_doppler.put(end_s - num_symbols)
    # print('I am number %d in process %d - finished' % (stream, getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('file_name_base', help='Name of the file inside the folders')
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    args = parser.parse_args()

    exp_dir = args.dir + '/'
    file_name_base = args.file_name_base + '.pcap'
    plotFlag = True

    # Wi-Fi link parameters
    n_ss = args.nss
    n_core = args.ncore
    n_tot = n_ss * n_core
    Tc = 6e-3
    fc = 5e9
    v_light = 3e8
    bw = 80
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)
    F_frequency = 256
    delta_f = 312.5E3
    T = 1 / delta_f
    frequency_vector_complete = np.zeros(F_frequency, )
    F_frequency_2 = F_frequency // 2
    for row in range(F_frequency // 2):
        freq_n = delta_f * (row - F_frequency / 2)
        frequency_vector_complete[row] = freq_n
        freq_p = delta_f * row
        frequency_vector_complete[row + F_frequency_2] = freq_p
    frequency_vector = np.delete(frequency_vector_complete, delete_idxs)
    frequency_vector_complete_hz = frequency_vector_complete + fc
    frequency_vector_hz = frequency_vector + fc
    F_frequency_retained = frequency_vector.shape[0]

    doppler_dim = 100
    window_length = 340
    stride_length = 1
    num_symbols = 41  # number of packet in a sample
    sliding = 1
    middle = int(mt.floor(num_symbols / 2))
    delta_v = round(v_light / (Tc * fc * num_symbols), 3)
    noise_lev = -3

    plot_interval = 0.08  # in seconds
    plot_int = int(plot_interval / Tc)

    num_elem_pred = int(2.2 / Tc)
    csi_doppler_matrix_array = np.zeros((n_tot, num_elem_pred, doppler_dim))

    path_estimation_interval = 0.05  # 0.1  # in seconds
    path_est_int = int(path_estimation_interval / Tc)

    subcarriers_space = 2
    delta_t = 5E-9
    range_considered = 5e-7
    idxs_range_considered = int(range_considered / delta_t + 1)
    # t_min = 0  # -T / 2  # -T/2
    # t_max = T / 3  # T/2
    t_min = 0
    t_max = 5E-7
    num_iteration_refinement = 10
    threshold = -3
    ToA_matrix, time_vector = optimization_utility.build_toa_matrix(frequency_vector_hz, delta_t, t_min, t_max)

    # Define the window layout
    my_new_theme = {'BACKGROUND': '#9eb9d4',
                    'TEXT': '#0C2C52',
                    'INPUT': '#c7e78b',
                    'TEXT_INPUT': '#000000',
                    'SCROLL': '#c7e78b',
                    'BUTTON': ('white', '#709053'),
                    'PROGRESS': ('#01826B', '#D0D0D0'),
                    'BORDER': 1,
                    'SLIDER_DEPTH': 0,
                    'PROGRESS_DEPTH': 0}
    sg.theme_add_new('MyNewTheme', my_new_theme)
    sg.theme('MyNewTheme')
    font1 = ('Arial', 30)
    font2 = ('Arial', 24)

    # Create plots
    seq_num = 1
    queues_plot = []
    processes_plot = []
    for stream in range(0, n_tot):
        queues_plot.append(Queue())
        p = Process(target=plots_antennas, args=(queues_plot[stream], stream))
        p.start()
        processes_plot.append(p)

    queues_plot_doppler = []
    processes_plot_doppler = []
    for stream in range(0, n_tot):
        queues_plot_doppler.append(Queue())
        p = Process(target=plots_doppler_antennas, args=(queues_plot_doppler[stream], stream))
        p.start()
        processes_plot_doppler.append(p)

    while True:
        file_name = exp_dir + str(seq_num) + '/' + file_name_base
        """
            File read csiread
        """
        file_name_next = exp_dir + str(seq_num+1) + '/'
        while not os.path.exists(file_name_next):
            time.sleep(0.01)

        # time.sleep(0.2)
        print("\n\n\n-----------------------------------------------")
        # print(file_name)
        print(seq_num)
        seq_num += 1
        try:
            csidata = csiread.Nexmon(file_name, chip='4366c0', bw=bw)
            csidata.read()
        except Exception:
            print('error in this packet, skipping...')
            continue

        csi_buff = csidata.csi

        """
            CSI_phase_sanitization_signal_preprocessing
        """
        csi_buff = np.fft.fftshift(csi_buff, axes=1)

        end_stream = int(np.floor(csi_buff.shape[0]/n_tot))

        processes = []
        with Manager() as manager:
            csi_doppler_matrix = manager.list()
            # figs = manager.list(figs)
            # axs = manager.list(axs)
            for stream in range(n_tot):

                signal_stream = csi_buff[stream:end_stream * n_tot:n_tot, :]

                p = Process(target=antenna_processing, args=(signal_stream, stream, csi_doppler_matrix,
                                                             queues_plot[stream], queues_plot_doppler[stream],
                                                             end_stream))
                processes.append(p)
                p.start()
                # pp = Process(target=plots_antennas, args=(signal_complete[:, :, i], i,
                #                                           figs[i], axs[i][0], axs[i][1]))
                # processes.append(pp)
                # pp.start()

            for p in processes:
                p.join()

            csi_doppler_matrix = np.asarray(csi_doppler_matrix)
            # discard the same quantity at the beginning and predict with old and new one
            csi_doppler_matrix_array[:, :num_elem_pred - end_stream + num_symbols, :] = \
                csi_doppler_matrix_array[:, end_stream - num_symbols:, :]
            csi_doppler_matrix_array[:, num_elem_pred - end_stream + num_symbols:, :] = csi_doppler_matrix

            a = 1
