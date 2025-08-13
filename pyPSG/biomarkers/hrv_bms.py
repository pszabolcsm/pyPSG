import numpy as np
import pandas as pd
from scipy.signal import welch, get_window
from scipy.interpolate import interp1d


## Time-domain metrics

def comp_AVNN(segment):

    """
    This function returns the mean RR interval (AVNN) over a segment of RR time series.
    
    :param segment: The input RR intervals time-series in seconds.
    :type segment: 1d-array
    :return: AVNN:  The mean RR interval over the segment.
    """
    
    segment = segment * 1000

    return np.mean(segment)

def comp_SDNN(segment):

    """
    This function returns the standard deviation over the RR intervals (SDNN) found in the input.
    
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return: SDNN:  The std. dev. over the RR intervals.
    """
    
    segment = segment * 1000

    return np.std(segment, ddof=1)

def comp_RMSSD(segment):

    """
    This function returns the square root of mean summed squares of RR interval differences.
        
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return: RMSSD:  square root of mean summed squares of NN interval differences.
    """
    
    segment = segment * 1000

    return np.sqrt(np.mean(np.diff(segment) ** 2))

def comp_PNN20(segment):

    """
    This function returns the percentage of the RR interval differences above .02 over a segment of RR time series.
    
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return PNN20:  The percentage of the RR interval differences above .02.
    """
    
    segment = segment * 1000

    return 100 * np.sum(np.abs(np.diff(segment)) > 20) / (len(segment) - 1)

def comp_PNN50(segment):

    """
    This function returns the percentage of the RR interval differences above .05 over a segment of RR time series.
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return PNN50:  The percentage of the RR interval differences above .05.
    """
    
    segment = segment * 1000

    return 100 * np.sum(np.abs(np.diff(segment)) > 50) / (len(segment) - 1)

def comp_SEM(segment):
    """
    This function returns standard error of the mean NN interval length.
    
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return: SEM: Standard error of the mean NN interval
    """
    
    segment = segment * 1000
    
    return  np.std(segment, ddof=1) / np.sqrt(len(segment))

## Non-linear metrics

def comp_poincare(segment):
    """
    Calculates HRV metrics from a Poincaré plot of the input data.

    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return:sd1: Standard deviation of RR intervals along the axis perpendicular to the line of identity.
            sd2: Standard deviation of RR intervals along the line of identity.
    """
    x_old = segment[:-1]
    y_old = segment[1:]
    alpha = -np.pi / 4
    rotation_matrix = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rri_rotated = np.dot(rotation_matrix(alpha), np.array([x_old, y_old]))
    x_new, y_new = rri_rotated
    # sd1 = np.std(y_new, ddof=0)
    #     sd2 = np.std(x_new, ddof=0)
    sd1 = np.std(y_new, ddof=1) * 1000
    sd2 = np.std(x_new, ddof=1) * 1000
    return sd1, sd2


def comp_SD1(segment):
    """
    Calculates the standard deviation of RR intervals along the axis perpendicular to the line of identity.
    
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return: SD1: standard deviation of RR intervals along the axis perpendicular to the line of identity
    """
    return comp_poincare(segment)[0]


def comp_SD2(segment):
    """
    Calculates the standard deviation of RR intervals along the line of identity..
    
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return: SD1: standard deviation of RR intervals along the line of identity
    """
    return comp_poincare(segment)[1]


def comp_DFA(segment, n_min=4, n_max=64, n_incr=2, alpha1_range=(4, 15), alpha2_range=(16, 64)):
    """
    Calculates the DFA (detrended fluctuation analysis) of a signal and it's scaling exponents.
    
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :param n_min: Minimal DFA block-size (default 4)
    :type n_min: int, optional
    :param n_max: Maximal DFA block-size (default 64)
    :type n_min: int, optional
    :param n_incr: increment value for n (default 2). Can also be less than 1, in which case we interpret it as the ratio of a geometric series on box sizes (n). This should produce box size values identical to the PhysioNet DFA implmentation.
    :type n_incr: int, optional
    :param alpha1_range: Range of block size values to use for calculating the alpha_1 scaling exponent. Default: [4, 15].
    :type alpha1_range: tuple, optional
    :param alpha2_range: Range of block size values to use for calculating the alpha_2 scaling exponent. Default: [16, 64].
    :type alpha2_range: tuple, optional

    Return: alpha1: Log-log slope of DFA in the low-scale region, alpha2: Log-log slope of DFA in the high-scale region

    """
    # Calculate zero-based interval time axis
    segment = np.asarray(segment).flatten()
    tnn = np.concatenate(([0], np.cumsum(segment[:-1])))
    
    # Integrate the signal without mean
    nni_int = np.cumsum(segment - np.mean(segment))
    N = len(nni_int)
    
    # Create n-axis (box-sizes)
    # If n_incr is less than 1 we interpret it as the ratio of a geometric series of boxis
    # This should produce box sizes identical to the Physionet DFA implementation
    if n_incr < 1:
        M = int(np.log2(n_max / n_min) * (1 / n_incr))
        n = np.unique(np.floor(n_min * (2 ** n_incr) ** np.arange(0, M + 1) + 0.5).astype(int))
    else:
        n = np.arange(n_min, n_max + 1, n_incr)
        
    # Initialize the array to store F(n) values
    fn = np.full(len(n), np.nan)
    
    for idx, nn in enumerate(n):
        #Calculate the number of windows we need for the current n
        num_win = N // nn
        
        # Break the signal into num_windows of n samples each
        sig_windows = np.reshape(nni_int[:nn * num_win], (nn, num_win), order='F')
        t_windows = np.reshape(tnn[:nn * num_win], (nn, num_win), order='F')
        sig_regressed = np.zeros_like(sig_windows)
        
        # Perform linear  regression in each window
        for ii in range(num_win):
            y = sig_windows[:, ii]
            X = np.column_stack((np.ones(nn), t_windows[:, ii]))
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            yn = X @ beta
            sig_regressed[:, ii] = yn
        
        # Calculate F(n), the value of the DFA for the current n
        fn[idx] = np.sqrt(np.sum((sig_windows - sig_regressed) ** 2) / N)
        
    # If fn is zero somewhere (might happen in the small scales if there's not enough data points there)
    # set it to some small constant to prevent log(0)=-Inf
    fn[fn < 1e-9] = 1e-9
    
    # Find DFA values in each of the alpha ranges
    alpha1_idx = (n >= alpha1_range[0]) & (n <= alpha1_range[1])
    alpha2_idx = (n >= alpha2_range[0]) & (n <= alpha2_range[1])
    
    # Find the line to the log-log DFA in each alpha range
    fn_log = np.log10(fn)
    n_log = np.log10(n)
    fit_alpha1 = np.polyfit(n_log[alpha1_idx], fn_log[alpha1_idx], 1)
    fit_alpha2 = np.polyfit(n_log[alpha2_idx], fn_log[alpha2_idx], 1)
    
    # Save the slopes of the lines
    alpha1 = fit_alpha1[0]
    alpha2 = fit_alpha2[0]
        
    return alpha1, alpha2, n, fn

def comp_alpha_1(segment):
    """
    Calculates the log-log slope of DFA in the low-scale region
    
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return: alpha1: log-log slope of DFA in the low-scale region

    """
    return comp_DFA(segment)[0]

def comp_alpha_2(segment):
    """
    Calculates the log-log slope of DFA in the low-scale region
    
    :param segment: The input RR intervals time-series.
    :type segment: 1d-array
    :return: alpha2: log-log slope of DFA in the high-scale region

    """
    return comp_DFA(segment)[1]

def buffer(X, n, p=0, opt=None):
    """Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html
    
    :param x: signal array
    :type x: ndarray
    :param n: number of data segments
    :type n: int
    :param p: number of values to overlap
    :type p: int
    :param opt: Initial condition options. default sets the first `p` values to zero,
                while 'nodelay' begins filling the buffer immediately.
    :return: buffer array created from x ((n,n)ndarray)
    """

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(X):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = X[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), X[:n-p]])
                i = n-p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = X[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[:,-1][-p:], col])
        i += n-p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n-len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result

def comp_sample_entropy(segment, m, r):
    """
    Calculate sample entropy (SampEn) of a signal.

    Sample entropy is a measure of the irregularity of a signal.

    :param segment: The input signal.
    :type segment: array_like
    :param m: Template length in samples.
    :type m: int
    :param r: Threshold for matching sample values.
    :type r: float

    :returns: The sample entropy value of the input signal.
    :rtype: float

    """
    N = len(segment)
    
    # Validations
    if m < 0 or r < 0:
        raise ValueError("Invalid parameter values")
    
    # Initialize template match counters
    A = 0
    B = 0
    if m == 0:
        B = N * (N - 1) / 2
    
    # Convert to float for speed, since this algorithm requires alot of in_memory copying
    segment = segment.astype(np.float32)
    
    # Create a matrix containing all templates (windows) of length m+1 (with m samples overlap) that exist in the signal; each row is a window
    templates_mat = buffer(segment, m + 1, m, opt='nodelay').T
    num_templates = templates_mat.shape[0]
    
    next_templates_mat_m = templates_mat[:, :m]
    next_templates_mat_1 = templates_mat[:, m]
    del templates_mat
    
    # Loop over all templates, calculating the Chedyshev distance between the current template and all following templates
    for win_idx in range(num_templates):
        # Extract the current template and all the templates following it
        curr_template_m = next_templates_mat_m[0, :]
        next_templates_mat_m = next_templates_mat_m[1:, :]
        
        curr_template_1 = next_templates_mat_1[0]
        next_templates_mat_1 = next_templates_mat_1[1:]
        
        # Calculate the absolute difference netween the current template and the each of the following templates
        diff_m = np.abs(next_templates_mat_m - curr_template_m)
        diff_1 = np.abs(next_templates_mat_1 - curr_template_1)
        
        # Calcualte Chebysehev distance: this is the max component of the absolute difference vektor
        # We will calculate two distances:
        # dist_B the Chebyshev distance using only the first m components
        dist_B = np.max(diff_m, axis=1) # max val of each row in diff_m
        
        # dist_A the max diff component (Chebyshev distance) using all m+1 components
        if m != 0:
            # max between column m+1 and dist_B (which is the maximum of columns 1..m)
            dist_A = np.maximum(dist_B, diff_1)
            
            # A template match is a case where the Chebyshev distance between
            # the current template and one of the next templates is less than r
            # Count the number of matches of length m+1 and of length m we have, and increment the appropiate counters
            A += np.sum(dist_A < r)
            B += np.sum(dist_B < r)
        else:
            #In case m is zero, dist_B is empty and dist_A is simply the diff_mat
            A += np.sum(diff_1 < r)
            
    # Calculate the sample entropy value based on the number of template matches
    if A == 0 or B == 0:
        sampen = np.nan
    else:
        sampen = -np.log(A / B)
    
    return sampen
            


def comp_MSE(segment, normalize_std = True, mse_max_scale = 20, sampen_m = 2, sampen_r = 0.2, mse_metrics = False):
    """
    Calculates the Multiscale Entropy (MSE) of a signal.

    Multiscale Entropy is a measure of a signal's complexity. This function computes
    the Sample Entropy of the signal at various scales from 1 to `mse_max_scale`.
    At each scale, the signal is downsampled by averaging `scale` samples, and
    Sample Entropy is calculated for the resulting coarse-grained signal.

    :param sig: Signal to calculate MSE for.
    :type sig: array_like
    :param mse_max_scale: Maximal scale to calculate up to. Default is 20.
    :type mse_max_scale: int, optional
    :param sampen_r: The 'r' parameter for Sample Entropy
        (maximum distance between matching points). Default is 0.2.
    :type sampen_r: float, optional
    :param sampen_m: The 'm' parameter for Sample Entropy
        (template length). Default is 2.
    :type sampen_m: int, optional
    :param normalize_std: Whether to normalize the signal to std=1 before entropy calculation.
        This affects the meaning of `r`.
    :type normalize_std: bool, optional
    :param plot: Whether to generate a plot of the results. Defaults to True if no output is returned.
    :type plot: bool, optional

    :returns:

    - **mse_result** (*ndarray*): The Sample Entropy value at each scale.
    - **scale_axis** (*ndarray*): The scale values corresponding to each MSE value.
    """

    # Normalize input
    N = len(segment)
    sig_normalized = segment - np.mean(segment)
    if normalize_std:
        sig_normalized = sig_normalized / np.sqrt(np.var(sig_normalized))
        
    # Preallocate results vector
    mse_result = np.zeros(mse_max_scale)
    
    scale_axis = np.arange(1, mse_max_scale + 1)
    
    for scale in scale_axis:
        # Split the signal into windows of length 'scale'
        max_idx = (N // scale) * scale
        sig_windows = np.reshape(sig_normalized[:max_idx], (scale, -1), order='F')
        
        # Calculate the mean of each window to obtain the 'coarse-grained' signal
        sig_coarse = np.mean(sig_windows, axis=0)
        
        # Calculate sample entropy of the coarse-grained signal
        sampen = comp_sample_entropy(sig_coarse, sampen_m, sampen_r)
        
        # If SampEn is Inf, replace with NaN
        if np.isinf(sampen):
            sampen = np.nan
        
        mse_result[scale - 1] = sampen
        
    # The first MSE value is the sample entropy
    
    if not mse_metrics:
        return mse_result[0]
    else:
        return mse_result

## Fragmentation metrics

def fragmentation_metrics(segment):
    """
    Compute fragmentation-related features from an NN interval segment.

    Detects inflection points based on changes in the sign of successive differences,
    and computes the segment lengths between them.

    :param segment: 1D array of NN intervals (e.g., RR intervals).
    :type segment: ndarray

    :returns:
        - **N** (*int*): Total number of NN intervals.
        - **ip** (*ndarray*): Binary array indicating inflection points (1 where inflection occurs).
        - **segment_lengths** (*ndarray*): Lengths of segments between inflection points.
    :rtype: tuple
    """
    N = len(segment)
    nni = segment.reshape(1, -1)  # reshape input into a row vector
    dnni = np.diff(nni)  # delta NNi: differences of conseccutive NN intervals
    ddnni = np.multiply(dnni[0, :-1], dnni[0, 1:])  # product of consecutive NN interval differences
    dd = np.asarray([-1] + list(ddnni) + [-1])

    # Logical vector of inflection point locations (zero crossings). Add a fake inflection points at the
    # beginning and end so that we can count the first and last segments (i.e. we want these segments
    # to be surrounded by inflection points like regular segments are).
    ip = (dd < 0).astype(int)
    ip_idx = np.where(ip)  # indices of inflection points
    segment_lengths = np.diff(ip_idx)[0]
    return N, ip, segment_lengths

def comp_PIP(segment):
    """
    Compute the Percentage of Inflection Points (PIP) in an NN interval segment.

    An inflection point is defined where the delta NN interval changes sign.
    Fake points are added at the beginning and end to enclose edge segments.

    :param segment: 1D array of NN intervals.
    :type segment: ndarray

    :returns: PIP – Percentage of inflection points in the segment.
    :rtype: float
    """

    N, ip, segment_lengths = fragmentation_metrics(segment)
    #Number of inflection points (where delta NNi changes sign). Subtract 2 for the fake points we added.
    nip = np.count_nonzero(ip)-2
    pip = nip/N     # percentage of inflection points (PIP)
    PIP = pip * 100
    return PIP

def comp_IALS(segment):
    """
    Compute the Inverse Average Length of Segments (IALS) in an NN interval segment.

    The segments are defined between inflection points.

    :param segment: 1D array of NN intervals.
    :type segment: ndarray

    :returns: IALS – Inverse of the mean segment length.
    :rtype: float
    """
    
    N, ip, segment_lengths = fragmentation_metrics(segment)
    IALS = 1 / np.mean(segment_lengths)  # Inverse Average Length of Segments (IALS)
    return IALS

def comp_PSS(segment):
    """
    Compute the Percentage of Short Segments (PSS) in an NN interval segment.

    Short segments are defined as segments with fewer than 3 NN intervals.

    :param segment: 1D array of NN intervals.
    :type segment: ndarray

    :returns: PSS – Percentage of NN intervals that belong to short segments.
    :rtype: float
    """
    
    N, ip, segment_lengths = fragmentation_metrics(segment)
    short_segment_lengths = segment_lengths[segment_lengths < 3]
    nss = np.sum(short_segment_lengths)
    pss = nss/N     # Percentage of NN intervals that are in short segments (PSS)
    PSS = pss * 100
    return PSS


def comp_PAS(segment):
    """
    Compute the Percentage of Alternating Segments (PAS) in an NN interval segment.

    Alternating segments are those where the segment length is > 3. This metric
    calculates the percentage of RR intervals that fall into such alternating segments.

    :param segment: 1D array of RR intervals.
    :type segment: ndarray

    :returns: PAS – Percentage of NN intervals in alternating segments of length > 3.
    :rtype: float
    """
    
    N, ip, segment_lengths = fragmentation_metrics(segment)
    alternation_segment_boundaries = np.asarray([1] + list((segment_lengths > 1).astype(int)) + [1])
    alternation_segment_lengths = np.diff(np.where(alternation_segment_boundaries))[0]
    # Percentage of NN intervals in alternation segments length > 3 (PAS)
    nas = np.sum(alternation_segment_lengths[alternation_segment_lengths > 3])
    pas = nas/N
    PAS = pas * 100
    return PAS

## Frequence-domain metrics

def freqband_power(pxx, f_axis, f_band):
    """
    Calculates the power in a frequency band.
    
    :param pxx: Power spectral density values.
    :type pxx: ndarray
    :param f_axis: Frequency axis corresponding to `pxx`.
    :type f_axis: ndarray
    :param f_band: Frequency band [f_low, f_high] over which to integrate the power.
    :type f_band: list or tuple or ndarray, length 2
    :returns: Power within the specified frequency band.
    :rtype: float

    """
    # Validate input
    if pxx.ndim != 1 or f_axis.ndim != 1:
        raise ValueError('pxx and f_axis must be 1D vectors')
    if len(pxx) != len(f_axis):
        raise ValueError('pxx and f_axis must have matching lengths')
    if not (isinstance(f_band, (list, tuple, np.ndarray)) and len(f_band) == 2):
        raise ValueError('f_band must be a 2-element array')
    if f_band[0] >= f_band[1]:
        raise ValueError('f_band width must be positive')
    
    # Convert to columns for consistency
    pxx = np.asarray(pxx).flatten()
    f_axis = np.asarray(f_axis).flatten()
    
    # Linearly interpolate the value of pxx at freq band limits
    interp_func = interp1d(f_axis, pxx, kind='linear', fill_value='extrapolate')
    pxx_f_band = interp_func(f_band)
    
    # Find the indices inside the band
    idx_band = (f_axis > f_band[0]) & (f_axis < f_band[1])
    
    # Create integration segment (the part of the signal we'll integrate over
    f_int = np.concatenate(([f_band[0]], f_axis[idx_band], [f_band[1]]))
    pxx_int = np.concatenate(([pxx_f_band[0]], pxx[idx_band], [pxx_f_band[1]]))
    
    # Integration using the trapezoidal method
    power = np.trapz(pxx_int, f_int)
    
    return power

def comp_freq(segment, vlf_band = [0.003, 0.04], lf_band = [0.04,  0.15], hf_band = [0.15,  0.4], resample_factor = 2.25, freq_osf=4, welch_overlap = 50, window_minutes = 5):
    """
    Compute frequency-domain HRV metrics using resampled RR interval data and Welch's method.
    This function estimates the PSD (power spectral density) of a given nn-interval sequence,
    and calculates the power in various frequency bands.
    
    :param segment: Sequence of NN intervals in seconds.
    :type segment: ndarray
    :param vlf_band: Frequency band for Very Low Frequency (VLF) power, default is [0.003, 0.04] Hz.
    :type vlf_band: list, optional
    :param lf_band: Frequency band for Low Frequency (LF) power, default is [0.04, 0.15] Hz.
    :type lf_band: list, optional
    :param hf_band: Frequency band for High Frequency (HF) power, default is [0.15, 0.4] Hz.
    :type hf_band: list, optional
    :param resample_factor: Multiplier for determining uniform resampling frequency (fs = resample_factor × max freq). Default is 2.25.
    :type resample_factor: float, optional
    :param freq_osf: Frequency oversampling factor to increase frequency resolution. Default is 4.
    :type freq_osf: int, optional
    :param welch_overlap: Overlap percentage between Welch windows (0–100). Default is 50.
    :type welch_overlap: int, optional
    :param window_minutes: Length of each Welch window in minutes. Default is 5.
    :type window_minutes: int, optional
    :returns:
        - **total_power** (*float*): Total spectral power across full frequency range.
        - **vlf_power** (*float*): Absolute power in VLF band.
        - **lf_power** (*float*): Absolute power in LF band.
        - **hf_power** (*float*): Absolute power in HF band.
        - **vlf_norm** (*float*): Normalized VLF power (% of total).
        - **lf_norm** (*float*): Normalized LF power (% of total).
        - **hf_norm** (*float*): Normalized HF power (% of total).
        - **lf_hf_ratio** (*float*): LF/HF power ratio.
    :rtype: tuple

    """
    # Calculate zero-based interval time axis
    segment = np.asarray(segment).flatten()
    tnn = np.concatenate(([0], np.cumsum(segment[:-1])))
    
    # Zero mean to removeDC component
    segment = segment - np.mean(segment)
    
    # window_minutes = max(1, int(np.floor((tnn[-1] - tnn[0]) / 60)))
    
    t_max = tnn[-1]
    f_min = vlf_band[0]
    f_max = hf_band[1]
    
    # Minimal window length (in seconds) needed to resolve f_min
    t_win_min = 1 / f_min
    
    # Increase windiw size if too small
    t_win = 60 * window_minutes
    if t_win < t_win_min:
        t_win = t_win_min
        
    # In case there's not enough data for one window, use entire signal length
    num_windows = int(np.floor(t_max / t_win))
    if num_windows < 1:
        num_windows = 1
        t_win = int(np.floor(tnn[-1] - tnn[0]))
        
    # Uniform sampling freq: take at least 2x more than f_max
    fs_uni = resample_factor * f_max  # Hz
    
    # Uniform time axis
    tnn_uni = np.arange(tnn[0], tnn[-1], 1 / fs_uni)
    n_win_uni = int(np.floor(t_win * fs_uni)) # Number of samples in each window
    num_windows_uni = int(np.floor(len(tnn_uni) / n_win_uni))
    
    # Build frequenceny axis
    ts = t_win / (n_win_uni - 1)  # Sampling time interval
    f_res = 1 / (n_win_uni * ts)  # Frequency resolution
    f_res = f_res / freq_osf  # Apply oversampling faktor
    
    
    f_axis = np.arange(f_res, f_max + f_res, f_res)
    f_axis = np.transpose(f_axis)
    
    # Check Nyquist criterion: we need at least 2*f_max*t_win samples in each window to resolve f_max
    if n_win_uni < 2 * f_max * t_win:
        print('Warning: Nyquist criterion not met for given window length and frequency bands')
        
    # Initialize output
    pxx_welch = np.zeros(len(f_axis))
    
    # Interpolate nn-intervals
    interp_func = interp1d(tnn, segment, kind='cubic')
    segment_uni = interp_func(tnn_uni)
    
    # Welch method
    window = get_window('hamming', n_win_uni)
    welch_overlap_samples = int(np.floor(n_win_uni * welch_overlap / 100))
    # Calculate Welch PSD
    nfft = 2**13
    f_welch, pxx_welch = welch(segment_uni, fs=fs_uni, window=window, noverlap=welch_overlap_samples, nfft=nfft,  scaling='density')
    pxx_welch = np.interp(f_axis, f_welch, pxx_welch)
    pxx_welch = pxx_welch / 2
    pxx_welch = pxx_welch * (1 / np.mean(window)) # Gain correction
    
    # Get entire frequency range
    total_band = [f_axis[0], f_axis[-1]]
    
    # Absolute power in each band
    total_power = freqband_power(pxx_welch, f_axis, total_band) * 1e6
    vlf_power = freqband_power(pxx_welch, f_axis, vlf_band) * 1e6
    lf_power = freqband_power(pxx_welch, f_axis, lf_band) * 1e6
    hf_power = freqband_power(pxx_welch, f_axis, hf_band) * 1e6
    
    # Calculate normalized power in each band
    vlf_norm = 100 * vlf_power / total_power
    lf_norm = 100 * lf_power / total_power
    hf_norm = 100 * hf_power / total_power
    lf_hf_ratio = lf_power / hf_power
    
    return total_power, vlf_power, lf_power, hf_power, vlf_norm, lf_norm, hf_norm, lf_hf_ratio


def get_all_metrics(rr_intervals):
    """
    Calculate all metrics for given intervals.
    
    :param rr_intervals: The input RR intervals time-series in seconds.
    :type rr_intervals: 1d-array
    
    :return: A dictionary with all metrics for given intervals.


    """
    rr_intervals = np.asarray(rr_intervals, dtype=np.float64).flatten()
    
    AVNN = comp_AVNN(rr_intervals)
    SDNN = comp_SDNN(rr_intervals)
    RMSSD = comp_RMSSD(rr_intervals)
    PNN50 = comp_PNN50(rr_intervals)
    SEM = comp_SEM(rr_intervals)
    
    PIP = comp_PIP(rr_intervals)
    IALS = comp_IALS(rr_intervals)
    PSS = comp_PSS(rr_intervals)
    PAS = comp_PAS(rr_intervals)
    
    SD1 = comp_SD1(rr_intervals)
    SD2 = comp_SD2(rr_intervals)
    alpha_1 = comp_alpha_1(rr_intervals)
    alpha_2 = comp_alpha_2(rr_intervals)
    MSE = comp_MSE(rr_intervals)
    
    (
        total_power,
        vlf_power,
        lf_power,
        hf_power,
        vlf_norm,
        lf_norm,
        hf_norm,
        lf_hf_ratio
    ) = comp_freq(rr_intervals)
    
    hrv_metrics = {
        "AVNN": AVNN,
        "SDNN": SDNN,
        "RMSSD": RMSSD,
        "PNN50": PNN50,
        "SEM": SEM,
        
        "PIP": PIP,
        "IALS": IALS,
        "PSS": PSS,
        "PAS": PAS,
        
        "SD1": SD1,
        "SD2": SD2,
        "alpha_1": alpha_1,
        "alpha_2": alpha_2,
        "MSE": MSE,
        
        "TOTAL_POWER": total_power,
        "VLF_POWER": vlf_power,
        "LF_POWER": lf_power,
        "HF_POWER": hf_power,
        "VLF_NORM": vlf_norm,
        "LF_NORM": lf_norm,
        "HF_NORM": hf_norm,
        "LF_HF_RATIO": lf_hf_ratio
    }
    
    return hrv_metrics
     
    

if __name__ == "__main__":
    a = 0
    # edf_path = "sample.edf"
    # channels = ["SpO2", "Pleth", "EKG"]
    # patient_name = "Patient_1"
    #
    # signals = read_edf_signals(edf_path, channels)
    #
    # signal = signals["EKG"]["signal"]
    # fs = signals["EKG"]["fs"]
    #
    # pre = Pre.Preprocessing(signal, fs)
    #
    # # Notch filter the powerline:
    # filtered_signal = pre.notch(n_freq=50)  # 50 Hz for european powerline, 60 Hz for USA
    #
    # # Bandpass for baseline wander and high-frequency noise:
    # filtered_signal = Pre.Preprocessing(filtered_signal, fs).bpfilt()
    #
    # fp = Fp.FiducialPoints(signal, fs)
    #
    # r_peaks = fp.jqrs()  #TODO: epltd function error, txt