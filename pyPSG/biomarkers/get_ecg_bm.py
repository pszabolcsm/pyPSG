from pyPSG.utils import HiddenPrints
from pecg import Preprocessing as Pre
from pecg.ecg import FiducialPoints as Fp
from pecg.ecg import Biomarkers as Bm
from pyPSG.biomarkers.get_hrv_bm import get_hrv_biomarkers

def get_ecg_biomarkers(signal, fs, matlab_path, get_hrv = True):
    pre = Pre.Preprocessing(signal, fs)
    
    # Notch filter the powerline:
    filtered_signal = pre.notch(n_freq=50)  # 50 Hz for european powerline, 60 Hz for USA
    
    # Bandpass for baseline wander and high-frequency noise:
    filtered_signal = pre.bpfilt()
    
    fp = Fp.FiducialPoints(filtered_signal, fs)
    # Two different peak detector algorithms:
    with HiddenPrints():  # to avoid long verbose of the peak detector functions
        jqrs_peaks = fp.jqrs()
        xqrs_peaks = fp.xqrs()
    
    fiducials = fp.wavedet(matlab_path, peaks=jqrs_peaks)
    
    bm = Bm.Biomarkers(filtered_signal, fs, fiducials)
    ints, stat_i = bm.intervals()
    waves, stat_w = bm.waves()
    
    ecg_biomarker = {
        "ints": ints,
        "stat_i": stat_i,
        "waves": waves,
        "stat_w": stat_w,
    }
    
    if get_hrv:
        hrv_biomarker = get_hrv_biomarkers(jqrs_peaks, fs)
        
        combined_biomarkers = {
            "ecg": ecg_biomarker,
            "hrv": hrv_biomarker
        }
        
        return combined_biomarkers
    else:
        return ecg_biomarker