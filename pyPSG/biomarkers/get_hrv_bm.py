import numpy as np
from pyPSG.biomarkers import hrv_bms as hrv

def get_hrv_biomarkers(peaks, fs):
    """
           This function computes heart rate variability (HRV) biomarkers from peak indices.
           When applied to PPG signals, the function effectively computes Beat Rate Variability (BRV).

           :param peaks: Indices of heartbeat peaks in the signal in seconds(sample positions).
           :type peaks: array-like
           :param fs: Sampling frequency of the signal in Hz.
           :type fs: float

           :return: Dictionary of computed HRV metrics.
           """
    
    # Calculate RR intervals in seconds (differences between successive peaks)
    rr_intervals = np.diff(peaks) / fs
    
    # Extract all available HRV metrics from the RR intervals
    all_metrics = hrv.get_all_metrics(rr_intervals)
    
    return all_metrics