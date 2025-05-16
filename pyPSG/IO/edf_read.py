import pyedflib

def read_edf_signals(edf_path, channel_names):
    """Load raw data from EDF file.

    :param edf_path: .
    :type edf_path: str
    :param channel_names: .
    :type channel_names: list

    :returns:
    
    """
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