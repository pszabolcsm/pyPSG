import numpy as np
import pyedflib

from pyPPG.datahandling import save_data


def cut_edf(edf_path, output_path, start_time, duration): #Time is specified in sec
    """
            Extract a segment from an EDF file and save it as a new EDF file. Time is specified in sec.
  
            :param edf_path: Path to the input EDF file.
            :type edf_path: str
            :param output_path: Path where the new EDF file will be saved.
            :type output_path: str
            :param start_time: Start time of the segment to cut (in seconds).
            :type start_time: float
            :param duration: Duration of the segment to cut (in seconds).
            :type duration: float
  
            :return: None
            """

  # Open the EDF file
    with pyedflib.EdfReader(edf_path) as edf:
      num_signals = edf.signals_in_file  # Get the number of signals in the file

      # Get signal labels,unit and frequences
      signal_labels = edf.getSignalLabels()

      # Initialize signals, freqences, measurment units
      all_sig=[]
      sig_dims = []
      sig_freqs = []
      
      for i in range(num_signals):
        # Get signal frequence and unit
        freq = edf.getSampleFrequency(i)
        sig_freqs.append(freq)
        dim = edf.getPhysicalDimension(i)
        sig_dims.append(dim)
        
        # Get signals and slice them
        start_sample =int(start_time * sig_freqs[i])
        num_samples = int(duration * sig_freqs[i])
        signal = edf.readSignal(i, digital=False)
        signal = signal[start_sample:start_sample + num_samples]
        all_sig.append(signal)

      # Write
      with pyedflib.EdfWriter(output_path, num_signals, file_type=pyedflib.FILETYPE_EDFPLUS) as edf_new:

        #Set channel informations
        channel_info = []
        for i in range(num_signals):
          pmin=np.min(all_sig[i])
          pmax=np.max(all_sig[i])
          
          # Handle flat signals to ensure the physical range is non-zero to avoid error
          if "Off" in signal_labels[i] and pmax == pmin:
            pmin = pmax - 1

          elif pmax == pmin:
            pmax += 0.01


          channel_info.append({
                "label": signal_labels[i],
                "dimension": sig_dims[i],
                "sample_frequency": sig_freqs[i],
                "physical_min": pmin,
                "physical_max": pmax,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            })
        edf_new.setSignalHeaders(channel_info)

        # Write the signals into the EDF
        edf_new.writeSamples(all_sig)

      edf._close()
      edf_new.close()
      



if __name__ == "__main__":

  cut_edf("mesa-sleep-0006.edf", "../../sample2.edf", 12000, 1200)