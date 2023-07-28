
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
import numpy as np
import csiread
import matplotlib
import matplotlib.pyplot as plt
import glob
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    args = parser.parse_args()

    exp_dir = args.dir + '/'
    plotFlag = True

    # Wi-Fi link parameters
    n_ss = args.nss
    n_core = args.ncore
    n_tot = n_ss * n_core
    fc = 5e9
    bw = 80
    F_frequency = 256
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)

    search_pattern = os.path.join(exp_dir, '*.pcap')
    files = glob.glob(search_pattern)

    for file in files:
        if file[-6:-5] == "2":
            save_name = exp_dir + 'Synced/Antenna_Separated/' + 'A01_'
        elif file[-6:-5] == "3":
            save_name = exp_dir + 'Synced/Antenna_Separated/' + 'A02_'
        elif file[-6:-5] == "4":
            save_name = exp_dir + 'Synced/Antenna_Separated/' + 'A03_'
        elif file[-6:-5] == "5":
            save_name = exp_dir + 'Synced/Antenna_Separated/' + 'A04_'

        try:
            csidata = csiread.Nexmon(file, chip='4366c0', bw=bw)
            csidata.read()
        except Exception:
            print('error in this packet, skipping...')

        offset = csidata.group(n_core, n_ss)
        csi_buff = csidata.csi[offset]

        """
            CSI_phase_sanitization_signal_preprocessing
        """
        signal_stream = np.fft.fftshift(csi_buff, axes=3)
        signal_stream[:, :, :, 64:] = - signal_stream[:, :, :, 64:]

        end_stream = int(np.floor(csi_buff.shape[0]))

        signal_stream = np.delete(signal_stream, delete_idxs, axis=3)
        mean_signal = np.mean(np.abs(signal_stream), axis=3, keepdims=True)
        signal_stream = signal_stream / mean_signal

        delete_zeros_idxs = np.argwhere(np.sum(signal_stream, axis=0) == 0)[:, 0]
        signal_stream = np.delete(signal_stream, delete_zeros_idxs, axis=1)
        signal_stream = np.squeeze(signal_stream)
        signal_stream = signal_stream[:10000, :, :]

        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        for i in range(signal_stream.shape[0]):
            ax[0, 0].plot(abs(signal_stream[i, 0, :]))
            ax[0, 1].plot(abs(signal_stream[i, 1, :]))
            ax[1, 0].plot(abs(signal_stream[i, 2, :]))
            ax[1, 1].plot(abs(signal_stream[i, 3, :]))
            ax[0, 0].set_title('Monitor_' + file[-6:-5] + '_1')
            ax[0, 1].set_title('Monitor_' + file[-6:-5] + '_2')
            ax[1, 0].set_title('Monitor_' + file[-6:-5] + '_3')
            ax[1, 1].set_title('Monitor_' + file[-6:-5] + '_4')
            ax[0, 0].set_ylim([0, 2.5])
            ax[0, 1].set_ylim([0, 2.5])
            ax[1, 0].set_ylim([0, 2.5])
            ax[1, 1].set_ylim([0, 2.5])
            ax[0, 0].grid(True)
            ax[0, 1].grid(True)
            ax[1, 0].grid(True)
            ax[1, 1].grid(True)

        fig.tight_layout()
        plt.savefig(exp_dir + 'Synced/' + 'A0' + file[-6:-5] + '.png')
        plt.show()

        for i in range(signal_stream.shape[1]):
            np.save(save_name + str(i+1), signal_stream[:, i, :])

