import numpy as np
import matplotlib.pyplot as plt
from pyPSG.biomarkers.get_ecg_bm import get_ecg_biomarkers
from pyPSG.biomarkers.get_ppg_bm import get_ppg_biomarkers

def plot_raw_data(signals):
    """
        Plots raw physiological signals for the first 60 seconds of data.

        :param signals: Dictionary containing signal data for each channel.
                        Keys are channel names, values are dictionaries with keys:
                        - "signal": array-like, the raw signal values
                        - "fs": float, sampling frequency in Hz
                        - "unit": str, unit of the signal (e.g., 'mV', '%', 'au')
        :type signals: dict

        :return: None. Displays a matplotlib figure with one subplot per signal channel.
        :rtype: None
        """
    num_signals = len(signals)
    
    fig, axs = plt.subplots(nrows=num_signals, ncols=1, figsize=(8, 3 * num_signals), sharex=True)
    
    # Convert num_signals to vector
    if num_signals == 1:
        axs = [axs]
    
    for i, (name, data) in enumerate(signals.items(), 1):
        sig = np.array(data['signal'])
        fs = data['fs']
        short_idx = int(fs * 60)
        sig = sig[:short_idx]
        time = np.arange(len(sig)) / fs
        unit = data['unit']
        
        ax = axs[i-1]
        ax.plot(time, sig, color='k', linewidth=1)
        ax.set_title(name)
        ax.set_xlabel(f'time (s)')
        ax.set_ylabel(f'{unit}')
        

            
        
        

    ax.legend()
    plt.tight_layout()
    plt.show()
    
def plot_variability(ppg_signal, ppg_fs, ecg_signal, ecg_fs, matlab_path):
    """
        Computes and plots breath rate variability (BRV) from PPG and heart rate variability (HRV) from ECG.

        BRV is calculated from detected PPG peaks, HRV from detected ECG peaks.
        Both are computed in 60-second sliding windows, and plotted as BPM over time.

        :param ppg_signal: Raw PPG signal.
        :type ppg_signal: array-like
        :param ppg_fs: Sampling frequency of the PPG signal in Hz.
        :type ppg_fs: float
        :param ecg_signal: Raw ECG signal.
        :type ecg_signal: array-like
        :param ecg_fs: Sampling frequency of the ECG signal in Hz.
        :type ecg_fs: float
        :param matlab_path: Path to the MATLAB executable (required for ECG peak detection).
        :type matlab_path: str

        :return: None. Displays a matplotlib figure comparing BRV and HRV over time.
        :rtype: None
        """
    
    #BRV
    ppg_peaks = get_ppg_biomarkers(ppg_signal, ppg_fs, get_peaks_only=True)
    
    ppg_peaks=ppg_peaks/ppg_fs
    
    window_duration = 60 #seconds
    ppg_max = max(ppg_peaks)
    window_begin = 0
    ppg_bpm = []
    ppg_time  = []
    
    while window_begin + window_duration < ppg_max:
        count = sum(window_begin <= p < window_begin + window_duration for p in ppg_peaks)
        ppg_bpm.append(count)
        ppg_time.append((window_begin + window_duration)/60)
        window_begin += window_duration
    
    #HRV
    ecg_peaks = get_ecg_biomarkers(ecg_signal, ecg_fs, matlab_path, get_peaks_only=True)
    
    ecg_peaks = ecg_peaks / ecg_fs
    
    window_duration = 60  # seconds
    ecg_max = max(ecg_peaks)
    window_begin = 0
    ecg_bpm = []
    ecg_time = []
    
    while window_begin + window_duration < ecg_max:
        count = sum(window_begin <= p < window_begin + window_duration for p in ecg_peaks)
        ecg_bpm.append(count)
        ecg_time.append((window_begin + window_duration) / 60)
        window_begin += window_duration
        
        
        
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(ppg_time, ppg_bpm, marker='o', linestyle='-', label='BR')
    plt.plot(ecg_time, ecg_bpm, color='r', marker='o', linestyle='-', label='HR')
    plt.xlabel('Time (m)')
    plt.ylabel('BPM')
    plt.title('BRV/HRV')
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    # if ppg_name != '':
    #     peaks = get_ppg_biomarkers(signals[ppg_name]['signal'], signals[ppg_name]['fs'], get_peaks_only=True)
    #     rr_intervals = np.diff(peaks) / signals[ppg_name]['fs']
    #     short_idx = int(signals[ppg_name]['fs'] * 60)
    #     rr_intervals = rr_intervals[:short_idx]
    #     # time = np.arange(len(rr_intervals)) / signals[ppg_name]['fs']
    #     peak_diffs = np.diff(peaks[:short_idx])
    #     bpm = 60 / (peak_diffs[1:])
    #     ax.plot(peaks[:-2] / signals[ppg_name]['fs'], bpm, color='b', linewidth=1, label='BR')
    #
    # if ecg_name != '':
    #     matlab_path = r'C://Program Files//MATLAB//MATLAB Runtime//v910//runtime//win64'
    #     peaks = get_ecg_biomarkers(signals[ecg_name]['signal'], signals[ecg_name]['fs'], matlab_path=matlab_path,
    #                                get_peaks_only=True)
    #     rr_intervals = np.diff(peaks) / signals[ecg_name]['fs']
    #     short_idx = int(signals[ecg_name]['fs'] * 60)
    #     rr_intervals = rr_intervals[:short_idx]
    #     # time = np.arange(len(rr_intervals)) / signals[ecg_name]['fs']
    #     peak_diffs = np.diff(peaks[:short_idx])
    #     bpm = 60 / (peak_diffs[1:])
    #
    #     ax.plot(peaks[:-2] / signals[ecg_name]['fs'], bpm, color='r', linewidth=1, label='HR')