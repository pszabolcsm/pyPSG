import pyedflib

def read_edf_signals(edf_path, channel_names):
    """Load raw data from EDF file.

        :param edf_path: Path to the EDF file.
        :type edf_path: str
        :param channel_names: List of channel names to extract from the EDF file.
        :type channel_names: list[str]

        :return: A dictionary where each key is a channel name and the value is another dictionary with:
                 - 'signal': The signal data.
                 - 'fs': The sampling frequency (in Hz) of the signal.

        """
    # Open the EDF file
    with pyedflib.EdfReader(edf_path) as edf:
        # Retrieve all signal labels
        labels = edf.getSignalLabels()
        signals = {}
        # Loop through each requested channel name and retrieve its sampling frequency and raw signal data
        for name in channel_names:
            if name in labels:
                idx = labels.index(name)
                fs = edf.getSampleFrequency(idx)
                sig = edf.readSignal(idx)
                signals[name] = {'signal': sig, 'fs': fs}
    return signals