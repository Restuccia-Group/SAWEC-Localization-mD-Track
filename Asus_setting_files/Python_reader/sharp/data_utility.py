
"""
    Copyright (C) 2022 Francesca Meneghello
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import math as mt
import matplotlib.animation as animation


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times'
rcParams['text.usetex'] = 'true'
#rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
rcParams['font.size'] = 16


def convert_to_number(lab, csi_label_dict):
    lab_num = np.argwhere(np.asarray(csi_label_dict) == lab)[0][0]
    return lab_num


def create_windows(csi_antennas, sample_length, stride_length, remove_mean=False):
    csi_matrix_stride = []
    for i in range(len(csi_antennas)):
        csi_i = csi_antennas[i]
        len_csi = csi_i.shape[0]
        for ii in range(0, len_csi - sample_length, stride_length):
            csi_wind = csi_i[ii:ii + sample_length, :]
            if remove_mean:
                csi_mean = np.mean(csi_wind, axis=1, keepdims=True)
                csi_wind = csi_wind - csi_mean
            csi_matrix_stride.append(csi_wind)
    return csi_matrix_stride


def create_windows_fom_array(csi_antennas, sample_length, stride_length, remove_mean=False):
    csi_matrix_stride = []
    end_idx = csi_antennas.shape[1]
    for i in range(len(csi_antennas)):
        for ii in range(0, end_idx - sample_length + 1, stride_length):
            csi_wind = csi_antennas[i, ii:ii + sample_length, :]
            if remove_mean:
                csi_mean = np.mean(csi_wind, axis=1, keepdims=True)
                csi_wind = csi_wind - csi_mean
            csi_matrix_stride.append(csi_wind)
    return csi_matrix_stride, end_idx


def plt_spectrum(spectrum_list, name_plot, step=100):
    fig = plt.figure()
    gs = gridspec.GridSpec(len(spectrum_list), 1, figure=fig)
    ticks_x = np.arange(0, spectrum_list[0].shape[0], step)
    ax = []

    for p_i in range(len(spectrum_list)):
        ax1 = fig.add_subplot(gs[(p_i, 0)])
        plt1 = ax1.pcolormesh(spectrum_list[p_i].T, shading='gouraud', cmap='viridis', linewidth=0, rasterized=True)
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'sub-channel')
        ax1.set_xlabel(r'time [s]')
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(ticks_x * 6e-3)
        ax.append(ax1)

    for axi in ax:
        axi.label_outer()
    fig.set_size_inches(20, 10)
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_amplitude_phase(signal_complete, name_plot, step=100):
    fig = plt.figure()
    gs = gridspec.GridSpec(signal_complete.shape[2], 1, figure=fig)
    ax = []

    for p_i in range(signal_complete.shape[2]):
        ax1 = fig.add_subplot(gs[(p_i, 0)])
        plt1 = ax1.plot(abs(signal_complete[:, :, p_i]).T)
        ax.append(ax1)

    for axi in ax:
        axi.label_outer()
    fig.set_size_inches(20, 10)
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()


def plt_fft_doppler_activities(doppler_spectrum_list, antenna, csi_label_dict, sliding_lenght, delta_v, name_plot):
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(15, 3)
    heights = [1]
    ncols = len(doppler_spectrum_list)
    widths = [1] * ncols
    widths.append(0.5)
    gs = fig.add_gridspec(ncols=ncols+1, nrows=1, width_ratios=widths, height_ratios=heights)
    step = 20
    step_x = 5
    ax = []
    for a_i in range(ncols):
        act = doppler_spectrum_list[a_i][antenna]
        length_v = mt.floor(act.shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, act.shape[0] + 1, int(act.shape[0]/step_x))

        ax1 = fig.add_subplot(gs[(0, a_i)])
        plt1 = ax1.pcolormesh(act.T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
        plt1.set_edgecolor('face')
        ax1.set_ylabel(r'velocity [m/s]')
        ax1.set_xlabel(r'time [s]')
        ax1.set_yticks(ticks_y + 0.5)
        ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
        ax1.set_xticks(ticks_x)
        ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * 6e-3, 2))

        title_p = csi_label_dict[a_i]
        ax1.set_title(title_p)
        ax.append(ax1)

        if a_i == 4:
            cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.7])
            cbar1 = fig.colorbar(plt1, cbar_ax)
            cbar1.ax.set_ylabel('power [dB]')

    for axi in ax:
        axi.label_outer()
    plt.savefig(name_plot, bbox_inches='tight')
    plt.close()
