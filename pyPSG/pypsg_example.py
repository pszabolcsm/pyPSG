from pyPSG.IO.edf_read import read_edf_signals
from pyPSG.IO.data_handling import save_data
from pyPSG.IO.plot import plot_raw_data, plot_variability
from pyPSG.biomarkers.get_spo2_bm import get_spo2_biomarkers
from pyPSG.biomarkers.get_ecg_bm import get_ecg_biomarkers
from pyPSG.biomarkers.get_ppg_bm import get_ppg_biomarkers
from pyPSG.biomarkers.get_hrv_bm import get_hrv_biomarkers

def pypsg_example(edf_path, matlab_path, channels = {"ppg": "", "ecg": "", "spo2": ""}):
    """
    Demonstrates the usage of the pyPSG toolbox to read an EDF file, visualize raw signals,
    compute heart rate variability (HRV) and breath rate variability (BRV),
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
    # Delete unnamed channels
    for ch, name in channels.items():
        if name == "":
            del channels[ch]
        
    #Get the channel names
    if "ppg" in channels:
        ppg_name = channels["ppg"]
    if "ecg" in channels:
        ecg_name = channels["ecg"]
    if "spo2" in channels:
        spo2_name = channels["spo2"]
    
    #Read the edf file
    signals = read_edf_signals(edf_path, channels.values())
    
    #Plot raw data
    plot_raw_data(signals)
    
    #Plot Heart Rate Variability and BRV
    plot_variability(signals[ppg_name]['signal'], signals[ppg_name]['fs'],signals[ecg_name]['signal'], signals[ecg_name]['fs'], matlab_path)
    
    #Calculate the biomarkers for each signal
    extracted_bms = {}
    ppg_bm = get_ppg_biomarkers(signals[ppg_name]['signal'], signals[ppg_name]['fs'])
    extracted_bms['ppg'] = ppg_bm
    ecg_bm = get_ecg_biomarkers(signals[ecg_name]['signal'], signals[ecg_name]['fs'], matlab_path)
    extracted_bms['ecg'] = ecg_bm
    spo2_bm = get_spo2_biomarkers(signals[spo2_name]['signal'], signals[spo2_name]['fs'])
    extracted_bms['spo2'] = spo2_bm
    
    #Save data into a .mat file
    save_data(extracted_bms, 'temp_dir/biomarkers')