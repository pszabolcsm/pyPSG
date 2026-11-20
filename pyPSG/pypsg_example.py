from pyPSG.IO.edf_read import read_edf_signals
from pyPSG.IO.data_handling import save_data
from pyPSG.IO.plot import plot_raw_data, plot_variability
from pyPSG.biomarkers.get_spo2_bm import extract_biomarkers_per_signal
from pyPSG.biomarkers import hrv_bms as hrv
from pyPSG.biomarkers.get_pcg_bm import extract_heart_sound_features

import numpy as np
import pandas as pd
import os
import scipy.stats as sp

from pobm.prep import set_range, median_spo2

from dotmap import DotMap
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM

from pyPSG.utils import HiddenPrints, select_file
from pecg import Preprocessing as Pre
from pecg.ecg import FiducialPoints as Fp
from pecg.ecg import Biomarkers as Bm

import pyPCG as pcg
import pyPCG.preprocessing as pcg_preproc
import pyPCG.segment as pcg_sgm
import pyPCG.features as pcg_features

def pypsg_example(edf_path=None, matlab_path=None, channels = {"ppg": "", "ecg": "", "spo2": "", "pcg": ""}):
    """
    Demonstrates the usage of the pyPSG toolbox to read an EDF file, visualize raw signals,
    compute heart rate variability (HRV) and beat rate variability (BRV),
    and extract biomarkers for PPG, ECG, and SpO₂ signals.

    :param edf_path: Path to the EDF file containing the physiological signals.
    :type edf_path: str
    :param matlab_path: Path to the MATLAB executable (required for ECG fiducial point detection and HRV computation).
    :type matlab_path: str
    :param channels: Dictionary mapping signal types to EDF channel names.
                     Keys can include "ppg", "ecg", and "spo2".
                     Values are the corresponding channel names in the EDF file.
                     Empty string values indicate that a channel should be ignored.
    :type channels: dict

    :return: None. The function generates plots of the raw data and HRV/BRV,
             extracts biomarkers for the available channels, and saves them
             into a `.mat` file in the directory `temp_dir/biomarkers`.
    :rtype: None
    """
    # If no EDF path is provided, open a file selection dialog
    if edf_path is None or edf_path == "":
        edf_path = select_file(
            title="Select EDF file",
            filetypes=[("EDF files", "*.edf")]
        )

    # Validate selection
    if not edf_path:
        raise ValueError("No EDF file selected.")
    
    # If no MATLAB path is provided, ask user to select the runtime directory
    if matlab_path is None or matlab_path == "":
        from tkinter import filedialog
        matlab_path = filedialog.askdirectory(
            title="Select MATLAB Runtime directory (R2021a)"
        )

    if not matlab_path:
        raise ValueError("No MATLAB path provided.")
    
    # Delete unnamed channels
    to_delete =[]
    for ch, name in channels.items():
        if name == "":
            to_delete.append(ch)
    
    for ch in to_delete:
        del channels[ch]
        
    #Get the channel names
    if "ppg" in channels:
        ppg_name = channels["ppg"]
    else: ppg_name = ''
    if "ecg" in channels:
        ecg_name = channels["ecg"]
    else: ecg_name = ''
    if "spo2" in channels:
        spo2_name = channels["spo2"]
    else: spo2_name = ''
    if "pcg" in channels:
        pcg_name = channels["pcg"]
    else: pcg_name = ''
    
    #Read the edf file
    signals = read_edf_signals(edf_path, channels.values())
    
    #Plot raw data
    plot_raw_data(signals)
    
    #Plot Heart Rate Variability and BRV
    plot_variability(signals, ppg_name, ecg_name, matlab_path)

    extracted_bms = {}
    
    ## Spo2 ##

    if spo2_name != '':
        # Remove values lower than 50 and greater than 100
        spo2_signal = set_range(signals[spo2_name]['signal'])
        # Apply median filter to the SpO2 signal
        spo2_signal = median_spo2(spo2_signal, FilterLength=301)
        # Calculate the time signal
        time_signal = np.arange(0, len(spo2_signal)) / signals[spo2_name]['fs']
    
        spo2_biomarker = pd.DataFrame()
    
        time_begin = time_signal[0]
        time_end = time_signal[-1]
    
        # Compute biomarkers
        spo2_bm = extract_biomarkers_per_signal(signal = spo2_signal, patient = 'Patient 1', time_begin=time_begin, time_end=time_end)
        
        extracted_bms['spo2'] = spo2_bm
        # Save data to csv
        os.makedirs("temp_dir", exist_ok=True)
        spo2_bm.to_csv("temp_dir/spo2_summary.csv", index=False)

    ## PPG ##
    
    if ppg_name != '':
        
        # Wrap raw signal and metadata into a DotMap structure
        ppg_signal = DotMap()
        ppg_signal.v = signals[ppg_name]['signal']
        ppg_signal.fs = signals[ppg_name]['fs']
        ppg_signal.start_sig = 0
        ppg_signal.end_sig = len(signals[ppg_name]['signal'])
        ppg_signal.name = "custom_ppg"
        
        # Initialise the filters
        filtering = True
        fL = 0.5000001
        fH = 12
        order = 4
        sm_wins = {'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}
        prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)
        
        # Filter and calculate the PPG, PPG', PPG", and PPG'" signals
        ppg_signal.filtering = filtering
        ppg_signal.fL = fL
        ppg_signal.fH = fH
        ppg_signal.order = order
        ppg_signal.sm_wins = sm_wins
        ppg_signal.ppg, ppg_signal.vpg, ppg_signal.apg, ppg_signal.jpg = prep.get_signals(s=ppg_signal)
        
        # Initialise the correction for fiducial points
        correction = pd.DataFrame()
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction.loc[0, corr_on] = True
        ppg_signal.correction = correction
        
        ## Create a PPG class
        s = PPG(s=ppg_signal, check_ppg_len=True)
        
        ## Get Fiducial points
        # Initialise the fiducials package
        fpex = FP.FpCollection(s=s)
        
        # Extract fiducial points
        ppg_fiducials = fpex.get_fiducials(s=s)
        
        # Create a fiducials class
        fp = Fiducials(fp=ppg_fiducials)
        
        # Initialise the biomarkers package
        bmex = BM.BmCollection(s=s, fp=fp)
        
        # Extract biomarkers
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
        
        # Create a biomarkers class
        ppg_bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)
        
        extracted_bms['ppg'] = ppg_bm
        
        # Save data to csv
        ppg_stats_df = pd.concat(
            ppg_bm.bm_stats.values(),
            axis=1
        )
        
        ppg_stats_df.to_csv(
            "temp_dir/ppg_stats.csv"
        )
        
        ## BRV ##
        
        ppg_peaks = ppg_fiducials.sp
        
        # Calculate RR intervals in seconds (differences between successive peaks)
        rr_intervals = np.diff(ppg_peaks) / signals[ppg_name]['fs']
        
        # Extract all available BRV metrics from the RR intervals
        brv_bm = hrv.get_all_metrics(rr_intervals, 30)
        
        # Compute summary statistics for each biomarker
        brv_stats = {}
        
        for name, values in brv_bm.items():
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
            
            brv_stats[name] = stats
        
        # Convert the statistics dictionary into a DataFrame
        brv_df = pd.DataFrame(brv_stats).T
        
        # Save BRV biomarker statistics into a CSV file
        brv_df.to_csv(
            "temp_dir/brv_summary.csv"
        )
    
    
    ## ECG ##
    
    if ecg_name != '':
    
        pre = Pre.Preprocessing(signals[ecg_name]['signal'], signals[ecg_name]['fs'])
        
        # Notch filter the powerline:
        filtered_signal = pre.notch(n_freq=50)  # 50 Hz for european powerline, 60 Hz for USA
        
        # Bandpass for baseline wander and high-frequency noise:
        filtered_signal = Pre.Preprocessing(filtered_signal, signals[ecg_name]['fs']).bpfilt()
        
        fp = Fp.FiducialPoints(filtered_signal, signals[ecg_name]['fs'])
        # Two different peak detector algorithms:
        with HiddenPrints():  # to avoid long verbose of the peak detector functions
            jqrs_peaks = fp.jqrs()
            
        # Compute fiducials using Wavedet algorithm
        ecg_fiducials = fp.wavedet(matlab_path, peaks=jqrs_peaks)
        
        # Compute intervals and waves biomarkers
        bm = Bm.Biomarkers(filtered_signal, signals[ecg_name]['fs'], ecg_fiducials)
        ints, stat_i = bm.intervals()
        waves, stat_w = bm.waves()
        
        ecg_bm = {
            "ints": ints,
            "stat_i": stat_i,
            "waves": waves,
            "stat_w": stat_w,
        }
        
        extracted_bms['ecg'] = ecg_bm
        
        # Merge interval and wave statistics
        ecg_stats = stat_i | stat_w
        
        # Convert ECG statistics into a DataFrame
        ecg_df = pd.DataFrame(ecg_stats).T
        
        # Save ECG biomarker statistics into a CSV file
        ecg_df.to_csv(
            "temp_dir/ecg_summary.csv"
        )
    
    
        ## HRV ##
        
        # Calculate RR intervals in seconds (differences between successive peaks)
        rr_intervals = np.diff(jqrs_peaks) / signals[ecg_name]['fs']
        
        # Extract all available HRV metrics from the RR intervals
        hrv_bm = hrv.get_all_metrics(rr_intervals, 30)
        
        # Compute summary statistics for each biomarker
        hrv_stats = {}
        
        for name, values in hrv_bm.items():
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
            
            hrv_stats[name] = stats
        
        # Convert the statistics dictionary into a DataFrame
        hrv_df = pd.DataFrame(hrv_stats).T
        
        # Save HRV biomarker statistics into a CSV file
        hrv_df.to_csv(
            "temp_dir/hrv_summary.csv"
        )
    
    ## PCG ##

    if pcg_name != '':
        
        # 1. Create PCG signal object and normalize
        pcg_signal = pcg.pcg_signal(signals[pcg_name]['signal'], signals[pcg_name]['fs'])
        pcg_signal = pcg.normalize(pcg_signal)
        
        # 2. Preprocessing
        lp_signal = pcg_preproc.filter(pcg_signal, lp_order=6, lp_cutoff=100, filt_type="LP")
        bp_signal = pcg_preproc.filter(lp_signal, hp_order=6, hp_cutoff=20, filt_type="HP")
        
        denoise_signal = pcg_preproc.wt_denoise(bp_signal, th=0.2, wt_family="coif4", wt_level=5)
        
        env_signal = pcg_preproc.homomorphic(denoise_signal, filt_ord=6, filt_cutfreq=8)
        
        # 4. Detect s1 and s2 sounds
        
        peakvals, peaks = pcg_sgm.adv_peak(env_signal)
        s1, s2 = pcg_sgm.peak_sort_diff(peaks)
        
        s1_start, s1_end = pcg_sgm.segment_peaks(s1, env_signal, start_drop=0.8, end_drop=0.7)
        s2_start, s2_end = pcg_sgm.segment_peaks(s2, env_signal, start_drop=0.8, end_drop=0.7)
        
        # Check segmentation results
        if len(s1_start) == 0 or len(s2_start) == 0:
            raise ValueError("Segmentation did not detect both S1 and S2 sounds.")
        
        # 4. Extract S1 and S2 features
        s1_features = extract_heart_sound_features(s1_start, s1_end, denoise_signal, env_signal, 1024)
        
        s2_features = extract_heart_sound_features(s2_start, s2_end, denoise_signal, env_signal, 1024)
        
        pcg_bms = {"S1": s1_features, "S2": s2_features}
        
        extracted_bms['pcg'] = pcg_bms


    # Save data into a .mat file
    save_data(extracted_bms, 'temp_dir/biomarkers')