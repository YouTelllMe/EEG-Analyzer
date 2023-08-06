import mne
import os


def load_mff(path: str, channels: list[str]) -> mne.io.Raw:
    """
    Load EEG data in the .mff format into a mne.io.Raw object
    """
    assert os.path.isdir(path), f"{path} does not exist"
    raw_mff = mne.io.read_raw_egi(path)
    raw_mff = raw_mff.pick(channels)
    return raw_mff


def frequency_filter(raw: mne.io.Raw,
                     lowpass: int | None = None,
                     highpass: int | None = None
                     ) -> mne.io.Raw:
    """
    Filters MNE Raw data object by frequency.

    Params
    ======
    raw: Raw MNE data object
    lowpass: lower boundary for frequency
    highpass: higher boundary for frequency
    """
    raw.load_data()
    return raw.copy().filter(l_freq = lowpass, h_freq = highpass)


def epoch(raw: mne.io.Raw, 
             stim_channel: str = "STI 014", 
             events_key: dict[str, int] = None
             ) -> mne.Epochs:
    """
    Performs epoching on data based on stimulus events.

    Params
    ======
    raw: Raw MNE data object
    """
    events = mne.find_events(raw, stim_channel=stim_channel)
    epochs = mne.Epochs(raw, events, event_id = events_key)
    return epochs


def artifact_filtering():
    pass