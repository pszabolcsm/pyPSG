from pyPSG.IO.edf_read import read_edf_signals
from pyPSG.IO.data_handling import save_data
from pyPSG.biomarkers.get_spo2_bm import get_spo2_biomarkers
from pyPSG.biomarkers.get_ecg_bm import get_ecg_biomarkers
from pyPSG.biomarkers.get_ppg_bm import get_ppg_biomarkers
from pyPSG.biomarkers.get_hrv_bm import get_hrv_biomarkers

def biomarker_extractor(edf_path, matlab_path, channels = {"ppg": "", "ecg": "", "spo2": ""}):
    
    for ch, name in channels.items():
        if name == "":
            del channels[ch]
    
    signals = read_edf_signals(edf_path, channels.values())
    
    extracted_bms = {}
    
    for ch, name in channels.items():
        if ch == "ecg":
            exec(
                ch + "_bm = get_" + ch + "_biomarkers(signals['" + name + "']['signal'], signals['" + name + "']['fs'], matlab_path)")
        else:
            exec(
            ch + "_bm = get_" + ch + "_biomarkers(signals['" + name + "']['signal'], signals['" + name + "']['fs'])")
        
        extracted_bms[ch] = eval(ch + "_bm")
    
    
    
    return extracted_bms

if __name__ == "__main__":
    matlab_path = r'C://Program Files//MATLAB//MATLAB Runtime//v910//runtime//win64'
    
    channels = {"ppg": "Pleth", "ecg": "ECG1"}
    
    extracted_bms = biomarker_extractor("meas3.edf", matlab_path, channels)
    
    save_data(extracted_bms, "biomarkers03")
    
    print(extracted_bms)