from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd


def envelope(data, fs=2000, hib=20, lob=400, lop=10, axis=0):
    hi_band = hib / (fs / 2)
    lo_band = lob / (fs / 2)
    b, a = butter(4, Wn=[hi_band, lo_band], btype="bandpass", output="ba")
    filtered_data = lfilter(
        b, a, data, axis=axis
    )  # 2nd order bandpass butterworth filter
    filtered_data = abs(filtered_data)  # Rectify the signal
    lo_pass = lop / (fs / 2)
    b, a = butter(4, lo_pass, output="ba")  # create low-pass filter to get EMG envelope
    filtered_data = lfilter(b, a, filtered_data, axis=axis)
    return filtered_data


def standardize_emg(train_data, test_data):
    mean = np.mean(train_data, axis=(0, 1), keepdims=True)
    std = np.std(train_data, axis=(0, 1), keepdims=True)
    std[std == 0] = 1
    train_data_standardized = (train_data - mean) / std
    test_data_standardized = (test_data - mean) / std
    return train_data_standardized, test_data_standardized


def normalize_labels(train_labels, test_labels):
    min_val = np.min(train_labels)
    max_val = np.max(train_labels)
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    train_labels_normalized = (train_labels - min_val) / range_val
    test_labels_normalized = (test_labels - min_val) / range_val
    return train_labels_normalized, test_labels_normalized
