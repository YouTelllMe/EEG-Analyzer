import mne
import matplotlib.pyplot as plt
import os


def plot_epoch(epochs: mne.Epochs) -> None:
    """
    Plots an interactive windows of an mne.Epochs object.

    Param
    =====
    epochs: epochs data object
    """
    epochs.plot(n_epochs=10)
    plt.show()

def plot_data(raw: mne.io.Raw, 
                         duration: int = 60,
                         scalings: float | None = None, 
                         n_channels: int = 20,
                         lowpass: float | None = None, 
                         highpass: float | None = None, 
                        ) -> None:
    """
    Plots an interactive windows of an mne.io.Raw object.

    Param
    =====
    raw: Raw MNE data object
    duration: width of interactive window 
    scalings: amplitude scaling, increase to fit data into the plot
    channels: channels to display
    n_channels: the number of channels to display; height of interactive window
    lowpass: lower boundary for frequency
    highpass: higher boundary for frequency
    """
    raw.plot(duration=duration, 
             proj=False, 
             scalings=scalings, 
             lowpass=lowpass,
             highpass=highpass, 
             n_channels=min(len(raw.ch_names),n_channels))
    plt.show()


def plot_spectrum():
    pass