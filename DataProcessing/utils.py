import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def get_knee_angle_peaks(knee_angle):
    peaks, _ = find_peaks(knee_angle, height=5)
    return peaks


def segment_array_with_label_peaks(array, knee_angle, cycle_length=1000):
    peaks = get_knee_angle_peaks(knee_angle)
    GC_segments = []
    for i in range(len(peaks) - 1):
        GC_segments.append(array[peaks[i] : peaks[i + 1]])
    GC_segments = [
        interp1d(np.linspace(0, 1, segment.shape[0]), segment, axis=0)(
            np.linspace(0, 1, cycle_length)
        )
        for segment in GC_segments
    ]
    GC_segments = np.stack(GC_segments, axis=2)
    return GC_segments


def rank_values(arr):
    sorted_indices = np.argsort(arr)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(arr))
    return ranks
