import pyedflib
import numpy as np
import pandas as pd
from dotmap import DotMap

# POBM moduls
from pobm.obm.desat import DesaturationsMeasures, desat_embedding
from pobm.obm.complex import ComplexityMeasures
from pobm.obm.burden import HypoxicBurdenMeasures
from pobm.obm.general import OverallGeneralMeasures
from pobm.obm.periodicity import PRSAMeasures, PSDMeasures
from pobm.prep import set_range, median_spo2

# PyPPG moduls
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI

#PECG moduls
from pecg import Preprocessing as ecgPre
from pecg.ecg import FiducialPoints as ecgFp
from pecg.ecg import Biomarkers as ecgBm

#HRV moduls
import mhrv_module as hrv

from utils import HiddenPrints


def read_edf_signals(edf_path, channel_names):

    with pyedflib.EdfReader(edf_path) as edf:
        labels = edf.getSignalLabels()
        signals = {}
        for name in channel_names:
            if name in labels:
                idx = labels.index(name)
                fs = edf.getSampleFrequency(idx)
                sig = edf.readSignal(idx)
                signals[name] = {'signal': sig, 'fs': fs}
    return signals


def features_all_desat(signal, time_signal, ODI_Threshold=6, hard_threshold=88, relative=True, desat_max_length=14400):
    time_signal = np.array(time_signal)

    desat_class = DesaturationsMeasures(ODI_Threshold=ODI_Threshold, hard_threshold=hard_threshold, relative=relative,
                                        desat_max_length=desat_max_length)
    desat_class.compute(signal)

    begin_idx = desat_class.begin
    end_idx = desat_class.end

    desaturations, desaturation_valid, desaturation_length_all, desaturation_int_100_all, \
        desaturation_int_max_all, desaturation_depth_100_all, desaturation_depth_max_all, \
        desaturation_slope_all = desat_embedding(begin_idx, end_idx)
    time_spo2_array = np.array(range(len(signal)))

    starts = []
    for (i, desaturation) in enumerate(desaturations):
        starts.append(desaturation['Start'])
        desaturation_idx = (time_spo2_array >= desaturation['Start']) & (time_spo2_array <= desaturation['End'])

        if np.sum(desaturation_idx) == 0:
            continue
        signal = np.array(signal)

        desaturation_time = time_spo2_array[desaturation_idx]
        desaturation_spo2 = signal[desaturation_idx]
        desaturation_min = np.nanmin(desaturation_spo2)
        desaturation_max = np.nanmax(desaturation_spo2)

        desaturation_valid[i] = True
        desaturation_length_all[i] = desaturation['Duration']
        desaturation_int_100_all[i] = np.nansum(100 - desaturation_spo2)
        desaturation_int_max_all[i] = np.nansum(desaturation_max - desaturation_spo2)
        desaturation_depth_100_all[i] = 100 - desaturation_min
        desaturation_depth_max_all[i] = desaturation_max - desaturation_min

        desaturation_idx_max = np.where(desaturation_spo2 == desaturation_max)[0][0]
        desaturation_idx_min = np.where(desaturation_spo2 == desaturation_min)[0][-1]
        desaturation_idx_max_min = np.arange(desaturation_idx_max, desaturation_idx_min + 1)

        if len(desaturation_idx_max_min) > 0:
            p = np.polyfit(np.int64(desaturation_time[desaturation_idx_max_min]),
                           desaturation_spo2[desaturation_idx_max_min], 1)

            desaturation_slope_all[i] = p[0]

    begin_time = time_signal[begin_idx]
    end_time = time_signal[end_idx]

    desat_patient = pd.DataFrame({
        "begin": begin_time,
        "end": end_time,
        "begin_idx": begin_idx,
        "end_idx": end_idx,
        "depth": desaturation_depth_max_all,
        "length": desaturation_length_all,
        "area": desaturation_int_max_all
    })
    return desat_patient


def extract_biomarkers_per_signal(signal, patient, time_begin, time_end):
    complexity_class = ComplexityMeasures()
    results_complexity = complexity_class.compute(signal)

    desat_class = DesaturationsMeasures()
    results_desat = desat_class.compute(signal)

    hypoxic_class = HypoxicBurdenMeasures(results_desat.begin, results_desat.end)
    results_hypoxic = hypoxic_class.compute(signal)

    statistics_class = OverallGeneralMeasures()
    results_overall = statistics_class.compute(signal)

    prsa_class = PRSAMeasures()
    results_PRSA = prsa_class.compute(signal)

    psd_class = PSDMeasures()
    results_PSD = psd_class.compute(signal)

    biomarkers = pd.DataFrame({
        "time begin": [time_begin],
        "time end": [time_end],
        "patient": [patient],
        "AV": [results_overall.AV],
        "MED": [results_overall.MED],
        "Min": [results_overall.Min],
        "SD": [results_overall.SD],
        "RG": [results_overall.RG],
        "P": [results_overall.P],
        "M": [results_overall.M],
        "ZC": [results_overall.ZC],
        "DI": [results_overall.DI],

        "ODI": [results_desat.ODI],

        "DL_u": [results_desat.DL_u],
        "DL_sd": [results_desat.DL_sd],
        "DA100_u": [results_desat.DA100_u],
        "DA100_sd": [results_desat.DA100_sd],
        "DAmax_u": [results_desat.DAmax_u],
        "DAmax_sd": [results_desat.DAmax_sd],
        "DD100_u": [results_desat.DD100_u],
        "DD100_sd": [results_desat.DD100_sd],
        "DDmax_u": [results_desat.DDmax_u],
        "DDmax_sd": [results_desat.DDmax_sd],
        "DS_u": [results_desat.DS_u],
        "DS_sd": [results_desat.DS_sd],
        "TD_u": [results_desat.TD_u],
        "TD_sd": [results_desat.TD_sd],

        "CA": [results_hypoxic.CA],
        "CT": [results_hypoxic.CT],
        "POD": [results_hypoxic.POD],
        "AODmax": [results_hypoxic.AODmax],
        "AOD100": [results_hypoxic.AOD100],

        "ApEn": [results_complexity.ApEn],
        "LZ": [results_complexity.LZ],
        "CTM": [results_complexity.CTM],
        "SampEn": [results_complexity.SampEn],
        "DFA": [results_complexity.DFA],

        "PRSAc": [results_PRSA.PRSAc],
        "PRSAad": [results_PRSA.PRSAad],
        "PRSAos": [results_PRSA.PRSAos],
        "PRSAsb": [results_PRSA.PRSAsb],
        "PRSAsa": [results_PRSA.PRSAsa],
        "AC": [results_PRSA.AC],

        "PSD_total": [results_PSD.PSD_total],
        "PSD_band": [results_PSD.PSD_band],
        "PSD_ratio": [results_PSD.PSD_ratio],
        "PSD_peak": [results_PSD.PSD_peak],
    })
    return biomarkers

def get_spo2_biomarkers(signal, fs, patient_name="Unknown"):
    spo2_signal = set_range(signal)
    spo2_signal = median_spo2(spo2_signal, FilterLength=301)
    time_signal = np.arange(0, len(spo2_signal)) / fs
    
    test_desat = features_all_desat(spo2_signal, time_signal, ODI_Threshold=4, hard_threshold=93, relative=False,
                                    desat_max_length=14400)
    
    biomarker = pd.DataFrame()
    
    time_begin = time_signal[0]
    time_end = time_signal[-1]
    
    biomarker = extract_biomarkers_per_signal(spo2_signal, patient_name, time_begin, time_end)
    
    return biomarker


def get_ppg_biomarkers(signal, fs, filtering=True, fL=0.5000001, fH=12, order=4, sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10}, correction=pd.DataFrame()):
    ppg_signal = DotMap()
    ppg_signal.v = signal
    ppg_signal.fs = fs
    ppg_signal.start_sig = 0
    ppg_signal.end_sig = len(signal)
    ppg_signal.name = "custom_ppg"
    
    # Initialise the filters
    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)
    
    # Filter and calculate the PPG, PPG', PPG", and PPG'" signals
    ppg_signal.filtering = filtering
    ppg_signal.fL = fL
    ppg_signal.fH = fH
    ppg_signal.order = order
    ppg_signal.sm_wins = sm_wins
    ppg_signal.ppg, ppg_signal.vpg, ppg_signal.apg, ppg_signal.jpg = prep.get_signals(s=ppg_signal)
    
    # Initialise the correction for fiducial points
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction.loc[0, corr_on] = True
    ppg_signal.correction = correction
    
    ## Create a PPG class
    s = PPG(s=ppg_signal, check_ppg_len=True)
    
    ## Get Fiducial points
    # Initialise the fiducials package
    fpex = FP.FpCollection(s=s)
    
    # Extract fiducial points
    fiducials = fpex.get_fiducials(s=s)
    # print("Fiducial points:\n", fiducials + s.start_sig) #TODO: szepen megcsinalani mint Marci
    
    # Create a fiducials class
    fp = Fiducials(fp=fiducials)
    
    # Calculate SQI
    # ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
    # print('Mean PPG SQI: ', ppgSQI, '%')
    
    # Initialise the biomarkers package
    fp = Fiducials(fp=fiducials)
    bmex = BM.BmCollection(s=s, fp=fp)
    
    # Extract biomarkers
    bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
    
    # Create a biomarkers class
    bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)
    
    return bm

def get_ecg_biomarkers(signal, fs, matlab_path):
    pre = ecgPre.Preprocessing(signal, fs)

    # Notch filter the powerline:
    filtered_signal = pre.notch(n_freq=50)  # 50 Hz for european powerline, 60 Hz for USA

    # Bandpass for baseline wander and high-frequency noise:
    filtered_signal = pre.bpfilt()

    fp = ecgFp.FiducialPoints(filtered_signal, fs)
    # Two different peak detector algorithms:
    with HiddenPrints():  # to avoid long verbose of the peak detector functions
        jqrs_peaks = fp.jqrs()
        xqrs_peaks = fp.xqrs()

    fiducials = fp.wavedet(matlab_path, peaks=jqrs_peaks)

    bm = ecgBm.Biomarkers(filtered_signal, fs, fiducials)
    ints, stat_i = bm.intervals()
    waves, stat_w = bm.waves()
    
    ecg_biomarker = {
        "ints": ints,
        "stat_i": stat_i,
        "waves": waves,
        "stat_w": stat_w,
    }

    return ecg_biomarker

def get_brv_metrics(signal, fs, filtering=True, fL=0.5000001, fH=12, order=4, sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10}, correction=pd.DataFrame()):
    ppg_signal = DotMap()
    ppg_signal.v = signal
    ppg_signal.fs = fs
    ppg_signal.start_sig = 0
    ppg_signal.end_sig = len(signal)
    ppg_signal.name = "custom_ppg"
    
    # Initialise the filters
    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm_wins)
    
    # Filter and calculate the PPG, PPG', PPG", and PPG'" signals
    ppg_signal.filtering = filtering
    ppg_signal.fL = fL
    ppg_signal.fH = fH
    ppg_signal.order = order
    ppg_signal.sm_wins = sm_wins
    ppg_signal.ppg, ppg_signal.vpg, ppg_signal.apg, ppg_signal.jpg = prep.get_signals(s=ppg_signal)
    
    # Initialise the correction for fiducial points
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction.loc[0, corr_on] = True
    ppg_signal.correction = correction
    
    ## Create a PPG class
    s = PPG(s=ppg_signal, check_ppg_len=True)
    
    ## Get Fiducial points
    # Initialise the fiducials package
    fpex = FP.FpCollection(s=s)
    
    # Extract fiducial points
    fiducials = fpex.get_fiducials(s=s)
    
    peaks = fiducials.sp
    
    pp_intervals = np.diff(peaks) / fs
    
    pp_intervals = np.array(pp_intervals, dtype=np.float64)
    
    metrics = hrv.get_all_metrics(pp_intervals)
    
    return metrics

def get_hrv_metrics(signal, fs):
    pre = ecgPre.Preprocessing(signal, fs)
    
    # Notch filter the powerline:
    filtered_signal = pre.notch(n_freq=50)  # 50 Hz for european powerline, 60 Hz for USA
    
    # Bandpass for baseline wander and high-frequency noise:
    filtered_signal = ecgPre.Preprocessing(filtered_signal, fs).bpfilt()
    
    fp = ecgFp.FiducialPoints(filtered_signal, fs)
    
    r_peaks = fp.jqrs()
    
    ecg_rr_intervals = np.diff(r_peaks) / fs
    
    all_metrics = hrv.get_all_metrics(ecg_rr_intervals)
    
    return all_metrics


if __name__ == "__main__":
    edf_path = "test03.edf"
    channels = ["SpO2", "Pleth", "EKG"]
    patient_name = "Patient_1"
    matlab_pat = r'C://Program Files//MATLAB//MATLAB Runtime//v910//runtime//win64'  # edit to the MATLAB Runtime path installed in your machine

    signals = read_edf_signals(edf_path, channels)
    
    ecg_signal = signals["EKG"]["signal"]
    ecg_fs = signals["EKG"]["fs"]
    ecg_results = get_ecg_biomarkers(ecg_signal, ecg_fs, matlab_pat)

    spo2_signal = signals["SpO2"]["signal"]
    spo2_fs = signals["SpO2"]["fs"]
    spo2_results = get_spo2_biomarkers(spo2_signal, spo2_fs, patient_name)

    ppg_signal = signals["Pleth"]["signal"]
    ppg_fs = signals["Pleth"]["fs"]
    ppg_results = get_ppg_biomarkers(ppg_signal, ppg_fs)
    
    hrv_results = get_hrv_metrics(ecg_signal, ecg_fs)
    brv_results = get_brv_metrics(ppg_signal, ppg_fs)
    
    print("SpO2 biomarkers")  #TODO: egy dataframebe a biomarkereket
    print(spo2_results)

    print("PPG biomarkers")
    print(ppg_results)
    
    print("ECG biomarkers")
    print(ecg_results)
    
    print("HRV biomarkers")
    print(ppg_results)
    
    print("BRV biomarkers")
    print(brv_results)
