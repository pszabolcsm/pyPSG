import numpy as np
import scipy.stats as sp
from pyPSG.biomarkers import hrv_bms as hrv
from pyPSG.IO.data_handling import save_data


def get_hrv_biomarkers(peaks, fs, win_len, include_last_partial):
    """
           This function computes heart rate variability (HRV) biomarkers from peak indices.
           When applied to PPG signals, the function effectively computes Beat Rate Variability (BRV).

           :param peaks: Indices of heartbeat peaks in the signal in seconds(sample positions).
           :type peaks: array-like
           :param fs: Sampling frequency of the signal in Hz.
           :type fs: float
           :param win_len: Window length in seconds.
           :type win_len: float

           :return: Dictionary of computed HRV metrics.
           """

    # Calculate RR intervals in seconds (differences between successive peaks)
    rr_intervals = np.diff(peaks) / fs

    # Extract all available HRV metrics from the RR intervals
    metric_vals = hrv.get_all_metrics(rr_intervals, win_len, include_last_partial)
    
    metric_stats = {}
    for name, values in metric_vals.items():
        stats = {}
        stats['mean'] = np.nanmean(values)
        stats['median'] = np.nanmedian(values)
        stats['std'] = np.nanstd(values)
        stats['percentile_25'] = np.nanpercentile(values, 25)
        stats['percentile_75'] = np.nanpercentile(values, 75)
        stats['iqr'] = sp.iqr(values, nan_policy='omit')
        stats['skew'] = sp.skew(values, nan_policy='omit')
        stats['kurtosis'] = sp.kurtosis(values, nan_policy='omit')
        stats['mad'] = sp.median_abs_deviation(values, nan_policy='omit')
        metric_stats[name] = stats
        
    all_metrics = {'vals': metric_vals, 'stats': metric_stats}
    
    return all_metrics