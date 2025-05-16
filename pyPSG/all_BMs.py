from pyPSG.IO.edf_read import read_edf_signals
from pyPSG.IO.data_handling import save_data

def biomarker_extractor(edf_path, matlab_path, channels = {"ppg": "", "ecg": "", "spo2": ""}): #TODO channel nevek dict-ben
    
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
    
    channels = {"ppg": "Pleth", "ecg": "EKG", "spo2": "SpO2"}
    
    extracted_bms = biomarker_extractor("../sample.edf", matlab_path, channels)
    
    save_data(extracted_bms, "biomarkers")
    
    print(extracted_bms)