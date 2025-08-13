import pandas as pd
from dotmap import DotMap

from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM

from pyPSG.biomarkers.get_hrv_bm import get_hrv_biomarkers

def get_ppg_biomarkers(signal, fs, filtering=True, fL=0.5000001, fH=12, order=4,
                       sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}, correction=pd.DataFrame(), get_brv = True, get_peaks_only = False):
    '''
        This function extract PPG-based biomarkers and optionally beat rate variability (BRV) from a PPG signal.

        :param signal: The raw PPG signal
        :type signal: array-like
        :param fs: The sampling frequency of the PPG signal
        :type fs: float
        :param get_brv: Whether to compute beat rate variability (BRV) biomarkers
        :type get_brv: bool, optional
        :param filtering: a bool for filtering
        :type filtering: bool
        :param fL: Lower cutoff frequency (Hz)
        :type fL: float
        :param fH: Upper cutoff frequency (Hz)
        :type fH: float
        :param order: Filter order
        :type order: int
        :param sm_wins: dictionary of smoothing windows in millisecond:
            - ppg: window for PPG signal
            - vpg: window for PPG' signal
            - apg: window for PPG" signal
            - jpg: window for PPG'" signal
        :type sm_wins: dict
        :param correction: DataFrame where the key is the name of the fiducial points and the value is bool
        :type correction: DataFrame
        :return: If `get_brv` is True, returns a dict with 'ppg' and 'brv' keys containing biomarkers.
                 Otherwise, returns only the PPG biomarkers object.
        '''
    
    # Wrap raw signal and metadata into a DotMap structure
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
    
    #Return peaks if get_peaks_only is true
    if get_peaks_only:
        peaks = fiducials.sp
        return peaks
    
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
    pyppg_bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)
    
    if get_brv:
        # Compute beat rate variability biomarkers (BRV)
        peaks = fiducials.sp
        brv_bm = get_hrv_biomarkers(peaks, fs)
        ppg_bm = {
            "ppg": pyppg_bm,
            "brv": brv_bm
        }
        
        return ppg_bm
    
    else:
        return pyppg_bm