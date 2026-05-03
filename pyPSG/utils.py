import os, sys
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

class HiddenPrints:
    """
        Context manager to temporarily suppress console output.
        
        """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def select_file(title="Select file", filetypes=[("EDF files", "*.edf")]):
    """
    Open a file dialog and return the selected file path.

    :param title: Dialog window title.
    :type title: str
    :param filetypes: List of (label, pattern) tuples defining selectable file types.
    :type filetypes: list of tuple
    :return: Selected file path, or empty string if cancelled.
    :rtype: str
    """

    # Initialize and hide the Tkinter root window
    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename(title=title, filetypes=filetypes)

def plot_leads(signals, fs, leads=range(12), labels=[''], peaks=None, figsize=(8,4)):
    """
        Plots multiple ECG leads from one or more signals with optional peak markers.

        :param signals: List of 2D numpy arrays, each representing an ECG signal (samples x leads).
        :type signals: list of np.ndarray
        :param fs: Sampling frequency of the signals in Hz.
        :type fs: float
        :param leads: Indices of leads to plot. Defaults to all 12 leads.
        :type leads: iterable of int, optional
        :param labels: List of labels for each signal, used in the legend.
        :type labels: list of str, optional
        :param peaks: Dictionary of peak positions to overlay on the plots, keys are labels and values are 2D arrays (samples x leads).
        :type peaks: dict, optional
        :param figsize: Figure size in inches (width, height).
        :type figsize: tuple, optional

        :return: None
        """
    t = np.linspace(0, signals[0].shape[0]/fs-1, signals[0].shape[0])
    fig, axes = plt.subplots(len(leads), 1, figsize=figsize)
    for ind, l in enumerate(leads):
        for s, signal_ in enumerate(signals):
            axes[ind].plot(t, signal_[:, l], label=labels[s])
        if peaks is not None:
            for key, value in peaks.items():
                peaks_i = value[:, l]
                axes[ind].plot(t[peaks_i], signal_[peaks_i, l], 'o', label=key)
    fig.supylabel('ECG (mV)')
    fig.supxlabel('Time (seconds)')
                      
    plt.legend(loc='upper right')
    plt.show()