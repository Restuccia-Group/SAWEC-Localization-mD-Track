import numpy as np


import numpy as np

def joint_aoa_tof_estimation(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, num_angles):
    matrix_cfr_aoa = np.dot(cfr_sample, aoa_matrix)
    matrix_cfr_aoa_toa = np.dot(toa_matrix, matrix_cfr_aoa) / (num_ant * num_subc)

    power_matrix_cfr_aoa_toa = np.abs(matrix_cfr_aoa_toa) ** 2

    index_max = np.argmax(power_matrix_cfr_aoa_toa)
    time_idx_max, angle_idx_max = divmod(index_max, num_angles)

    amplitudes_time_max = matrix_cfr_aoa_toa[time_idx_max, angle_idx_max]

    # You may uncomment the following lines for debugging or visualization
    # plt.figure()
    # plt.pcolor(np.abs(matrix_cfr_aoa).T)
    # plt.show()
    # plt.figure()
    # plt.pcolor(np.abs(power_matrix_cfr_aoa_toa).T)
    # plt.show()

    return amplitudes_time_max, power_matrix_cfr_aoa_toa[time_idx_max, angle_idx_max], time_idx_max, angle_idx_max

import numpy as np

def individual_aoa_tof_estimation(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, index_toa):
    cfr_sample_toa = np.dot(toa_matrix[index_toa, :], cfr_sample)

    matrix_cfr_aoa = np.dot(cfr_sample_toa, aoa_matrix)
    angle_idx_max = np.argmax(np.abs(matrix_cfr_aoa) ** 2)

    cfr_sample_aoa = np.dot(cfr_sample, aoa_matrix[:, angle_idx_max])
    matrix_cfr_toa = np.dot(toa_matrix, cfr_sample_aoa) / (num_ant * num_subc)

    power_matrix_cfr_toa = np.abs(matrix_cfr_toa) ** 2
    time_idx_max = np.argmax(power_matrix_cfr_toa)

    amplitudes_time_max = matrix_cfr_toa[time_idx_max]

    return amplitudes_time_max, power_matrix_cfr_toa[time_idx_max], time_idx_max, angle_idx_max


def md_track_2d(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, num_angles, num_iteration_refinement, threshold):
    paths = []
    paths_amplitude = []
    paths_power = []
    paths_toa = []
    paths_aoa = []
    cfr_sample_residual = cfr_sample

    # mD-Track INITIAL ESTIMATION
    while not paths_power or path_power > paths_power[0] * 10 ** threshold:
        path_amplitude, path_power, path_toa, path_aoa = joint_aoa_tof_estimation(cfr_sample_residual, aoa_matrix,
                                                                                  toa_matrix, num_ant, num_subc,
                                                                                  num_angles)

        paths_amplitude.append(path_amplitude)
        paths_power.append(path_power)
        paths_toa.append(path_toa)
        paths_aoa.append(path_aoa)

        signal_path = path_amplitude * \
                      np.conj(aoa_matrix[:, path_aoa]) * \
                      np.tile(np.expand_dims(np.conj(toa_matrix[path_toa, :]), -1), num_ant)
        paths.append(signal_path)

        cfr_sample_residual -= signal_path

    num_estimated_paths = len(paths)

    # mD-Track ITERATIVE REFINEMENT
    paths_refined_amplitude = []
    paths_refined_toa = []
    paths_refined_aoa = []

    for iteration in range(num_iteration_refinement):
        for path_idx in range(num_estimated_paths):
            cfr_single_path = paths[path_idx] + cfr_sample_residual
            path_amplitude, path_power, path_toa, path_aoa = individual_aoa_tof_estimation(cfr_single_path,
                                                                                           aoa_matrix, toa_matrix,
                                                                                           num_ant, num_subc,
                                                                                           paths_toa[path_idx])

            if iteration == num_iteration_refinement - 1:
                paths_refined_amplitude.append(path_amplitude)
                paths_refined_toa.append(path_toa)
                paths_refined_aoa.append(path_aoa)

            signal_path_refined = path_amplitude * \
                                  np.conj(aoa_matrix[:, path_aoa]) * \
                                  np.tile(np.expand_dims(np.conj(toa_matrix[path_toa, :]), -1), num_ant)

            paths[path_idx] = signal_path_refined

        cfr_cumulative_paths = sum(paths)
        cfr_sample_residual = cfr_sample - cfr_cumulative_paths

    return paths, paths_refined_amplitude, paths_refined_toa, paths_refined_aoa
