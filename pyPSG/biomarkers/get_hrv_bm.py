import numpy as np
from pyPSG.biomarkers import hrv_bms as hrv


def get_hrv_biomarkers(peaks, fs):
    rr_intervals = np.diff(peaks) / fs
    
    all_metrics = hrv.get_all_metrics(rr_intervals)
    
    return all_metrics