import numpy as np


def max_amplitude(emg_segment):
    # Note that the emg segment is of shape (cycle_length, combinations)
    max_amplitudes = np.max(emg_segment, axis=0)
    return max_amplitudes


def max_peak_location(emg_segment):
    max_indices = np.argmax(emg_segment, axis=0)
    return max_indices


def area_uder_the_curve(emg_segment):
    auc_per_column = np.trapz(emg_segment, axis=0)
    return auc_per_column


def calculate_CoV(features_across_cycles):
    mean = np.mean(features_across_cycles, axis=0)
    # Calculate the standard deviation along the columns
    std_dev = np.std(features_across_cycles, axis=0)
    # Calculate the CoV (standard deviation divided by the mean)
    cov = std_dev / mean
    return cov


def get_all_metrics(cycle_emg_data):
    # Here, the emg data is going to be a 3D array of shape (cycle_length, n_channels, n_cycles)
    max_val = []
    max_loc = []
    auc = []
    for i in range(cycle_emg_data.shape[-1]):
        max_val.append(np.expand_dims(max_amplitude(cycle_emg_data[:, :, i]), axis=0))
        max_loc.append(
            np.expand_dims(max_peak_location(cycle_emg_data[:, :, i]), axis=0)
        )
        auc.append(np.expand_dims(area_uder_the_curve(cycle_emg_data[:, :, i]), axis=0))
    max_val_cov = calculate_CoV(np.concatenate(max_val, axis=0))
    max_loc_cov = calculate_CoV(np.concatenate(max_loc, axis=0))
    auc_cov = calculate_CoV(np.concatenate(auc, axis=0))
    return (max_val_cov, max_loc_cov, auc_cov)
