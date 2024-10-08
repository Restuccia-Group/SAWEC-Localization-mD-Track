
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
import cmath as cmt
import osqp
import scipy
import math as mt


def convert_to_complex_osqp(real_im_n):
    len_vect = real_im_n.shape[0] // 2
    complex_n = real_im_n[:len_vect] + 1j * real_im_n[len_vect:]
    return complex_n


def build_T_matrix(frequency_vector, delta_t_, t_min_, t_max_):
    F_frequency = frequency_vector.shape[0]
    L_paths = int((t_max_ - t_min_) / delta_t_)
    T_matrix = np.zeros((F_frequency, L_paths), dtype=complex)
    time_matrix = np.zeros((L_paths,))
    for col in range(L_paths):
        time_col = t_min_ + delta_t_ * col
        time_matrix[col] = time_col
        for row in range(F_frequency):
            freq_n = frequency_vector[row]
            T_matrix[row, col] = cmt.exp(-1j * 2 * cmt.pi * freq_n * time_col)
    return T_matrix, time_matrix


def lasso_regression_osqp_fast(H_matrix_, T_matrix_, selected_subcarriers, row_T, col_T, Im, Onm, P, q, A2, A3,
                               ones_n_matr, zeros_n_matr, zeros_nm_matr):
    # time_start = time.time()
    T_matrix_selected = T_matrix_[selected_subcarriers, :]
    H_matrix_selected = H_matrix_[selected_subcarriers]

    T_matrix_real = np.zeros((2*row_T, 2*col_T))
    T_matrix_real[:row_T, :col_T] = np.real(T_matrix_selected)
    T_matrix_real[row_T:, col_T:] = np.real(T_matrix_selected)
    T_matrix_real[row_T:, :col_T] = np.imag(T_matrix_selected)
    T_matrix_real[:row_T, col_T:] = - np.imag(T_matrix_selected)

    H_matrix_real = np.zeros((2*row_T))
    H_matrix_real[:row_T] = np.real(H_matrix_selected)
    H_matrix_real[row_T:] = np.imag(H_matrix_selected)

    n = col_T*2

    # OSQP data
    A = scipy.sparse.vstack([scipy.sparse.hstack([T_matrix_real, -Im, Onm.T]),
                             A2,
                             A3], format='csc')
    l = np.hstack([H_matrix_real, - np.inf * ones_n_matr, zeros_n_matr])
    u = np.hstack([H_matrix_real, zeros_n_matr, np.inf * ones_n_matr])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

    # Update linear cost
    lambd = 1E-1
    q_new = np.hstack([zeros_nm_matr, lambd * ones_n_matr])
    prob.update(q=q_new)

    # Solve
    res = prob.solve()

    x_out = res.x
    x_out_cut = x_out[:n]

    r_opt = convert_to_complex_osqp(x_out_cut)
    return r_opt


def build_toa_matrix(frequency_vector_hz, delta_t_, t_min_, t_max_):
    l_paths = int(np.round((t_max_ - t_min_) / delta_t_))
    time_vector_ = np.linspace(t_min_, t_max_, l_paths)
    time_freq_matrix = np.expand_dims(time_vector_, -1) * np.expand_dims(frequency_vector_hz, 0)
    toa_matrix_ = np.exp(1j * 2 * mt.pi * time_freq_matrix)
    return toa_matrix_, time_vector_


def tof_estimation(cfr_sample, toa_matrix, num_subc):
    matrix_cfr_toa = np.dot(toa_matrix, cfr_sample) / num_subc

    power_matrix_cfr_toa = np.power(np.abs(matrix_cfr_toa), 2)
    # plt.figure()
    # plt.pcolormesh(power_matrix_cfr_toa)
    # plt.show()
    time_idx_max = np.argmax(power_matrix_cfr_toa)

    amplitudes_time_max = matrix_cfr_toa[time_idx_max]

    return amplitudes_time_max, power_matrix_cfr_toa[time_idx_max], time_idx_max


def md_track_2d(cfr_sample, toa_matrix, num_subc, num_iteration_refinement, threshold):
    # start = time.time()
    paths = []

    # mD-Track INITIAL ESTIMATION
    paths_amplitude = []
    paths_power = []
    paths_toa = []
    cfr_sample_residual = cfr_sample
    while True:
        path_amplitude, path_power, path_toa = tof_estimation(cfr_sample_residual, toa_matrix, num_subc)

        try:
            ref = paths_power[0]
            if path_power <= ref*10**threshold:
                break
        except IndexError:
            True  # it is the first path, keep it and go ahead

        paths_amplitude.append(path_amplitude)
        paths_power.append(path_power)
        paths_toa.append(path_toa)

        signal_path = path_amplitude * np.conj(toa_matrix[path_toa, :])
        paths.append(signal_path)

        cfr_sample_residual = cfr_sample_residual - signal_path

    num_estimated_paths = len(paths)

    # mD-Track ITERATIVE REFINEMENT
    paths_refined_amplitude = []
    paths_refined_toa = []
    for iteration in range(num_iteration_refinement):
        for path_idx in range(num_estimated_paths):
            cfr_single_path = paths[path_idx] + cfr_sample_residual
            path_amplitude, path_power, path_toa = tof_estimation(cfr_single_path, toa_matrix, num_subc)

            if iteration == num_iteration_refinement-1:
                paths_refined_amplitude.append(path_amplitude)
                paths_refined_toa.append(path_toa)

            signal_path_refined = path_amplitude * np.conj(toa_matrix[path_toa, :])

            paths[path_idx] = signal_path_refined  # update the path with the refinement
            cfr_cumulative_paths = sum(paths)
            cfr_sample_residual = cfr_sample - cfr_cumulative_paths

    # end = time.time()
    # print(end - start)

    return paths, paths_refined_amplitude, paths_refined_toa, num_estimated_paths, cfr_sample_residual
