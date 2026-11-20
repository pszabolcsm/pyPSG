from pathlib import Path

import numpy as np
import pyPCG as pcg
import pyPCG.preprocessing as preproc
import pyPCG.segment as sgm
import pyPCG.features as ftr
import pyPCG.io as pcg_io


def extract_heart_sound_features(starts, ends, signal, envelope, nfft):
    """Calculate features for individual heart sound segments."""

    features = {}

    # Time-domain features
    features["duration"] = ftr.time_delta(starts, ends, signal)
    features["onset"] = ftr.ramp_time(starts, ends, envelope)
    features["exit"] = ftr.ramp_time(starts, ends, envelope, type="exit")
    features["zero_cross_rate"] = ftr.zero_cross_rate(starts, ends, signal)
    features["peak_spread"] = ftr.peak_spread(starts, ends, envelope)
    features["peak_centroid"] = ftr.peak_centroid(starts, ends, envelope)
    features["peak_width"] = ftr.peak_width(starts, ends, envelope)

    # Frequency-domain features
    features["max_freq"] = ftr.max_freq(starts, ends, signal, nfft=nfft)
    features["spectral_spread"] = ftr.spectral_spread(starts, ends, signal, nfft=nfft)
    features["spectral_centroid"] = ftr.spectral_centroid(starts, ends, signal, nfft=nfft)
    features["spectral_width"] = ftr.spectral_width(starts, ends, signal, nfft=nfft)
    features["spectrum"] = ftr.spectrum_raw(starts, ends, signal, nfft=nfft)

    # Wavelet-based features
    features["max_cwt"] = ftr.max_cwt(starts, ends, signal)
    features["cwt_peakdist"] = ftr.cwt_peakdist(starts, ends, signal)
    features["dwt_intensity"] = ftr.dwt_intensity(starts, ends, signal)
    features["dwt_entropy"] = ftr.dwt_entropy(starts, ends, signal)

    # Complexity-based features
    features["katz_fd"] = ftr.katz_fd(starts, ends, signal)

    return features


def get_pcg_biomarkers(pcg_data, fs, lp_order=6, lp_cutoff=100, hp_order=6, hp_cutoff=20,
                       denoise_threshold=0.2, wavelet_family="coif4", wavelet_level=5, envelope_filter_order=6,
                       envelope_cutoff=8, nfft=1024):
    """
    Extract S1 and S2 features from a PCG signal.

    Parameters
    ----------
    pcg_data : numpy.ndarray
        Input PCG signal.
    fs : int
        Sampling frequency of the PCG signal in Hz.
    hsmm_path : str or pathlib.
        Path to the pretrained HSMM model.
    lp_order : int
        Order of the low-pass filter.
    lp_cutoff : float
        Cutoff frequency of the low-pass filter in Hz.
    hp_order : int
        Order of the high-pass filter.
    hp_cutoff : float
        Cutoff frequency of the high-pass filter in Hz.
    denoise_threshold : float
        Wavelet denoising threshold.
    wavelet_family : str
        Wavelet family used for denoising.
    wavelet_level : int
        Wavelet decomposition level.
    envelope_filter_order : int
        Filter order used for homomorphic envelope calculation.
    envelope_cutoff : float
        Low-pass cutoff frequency for homomorphic envelope calculation.
    nfft : int
        Number of FFT points used for spectral feature extraction.

    Returns
    -------
    dict
        Dictionary containing S1 and S2 feature dictionaries.
    """

    # 1. Create PCG signal object and normalize
    signal = pcg.pcg_signal(pcg_data, fs)
    signal = pcg.normalize(signal)

    # 2. Preprocessing
    lp_signal = preproc.filter(signal, lp_order, lp_cutoff, "LP")
    bp_signal = preproc.filter(lp_signal, hp_order, hp_cutoff, "HP")

    denoise_signal = preproc.wt_denoise(bp_signal, th=denoise_threshold, wt_family=wavelet_family, wt_level=wavelet_level)

    env_signal = preproc.homomorphic(denoise_signal, filt_ord=envelope_filter_order, filt_cutfreq=envelope_cutoff)

    # 4. Detect s1 and s2 sounds
    
    peakvals, peaks = sgm.adv_peak(env_signal)
    s1, s2 = sgm.peak_sort_diff(peaks)
    
    s1_start, s1_end = sgm.segment_peaks(s1, env_signal, start_drop=0.8, end_drop=0.7)
    s2_start, s2_end = sgm.segment_peaks(s2, env_signal, start_drop=0.8, end_drop=0.7)
    

    # Check segmentation results
    if len(s1_start) == 0 or len(s2_start) == 0:
        raise ValueError("HSMM segmentation did not detect both S1 and S2 sounds.")

    # 4. Extract S1 and S2 features
    s1_features = extract_heart_sound_features(
        s1_start, s1_end, denoise_signal, env_signal, nfft
    )

    s2_features = extract_heart_sound_features(
        s2_start, s2_end, denoise_signal, env_signal, nfft
    )

    # 5. Return extracted features
    return {"S1": s1_features, "S2": s2_features}