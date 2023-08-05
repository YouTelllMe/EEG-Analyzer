import mne
import os

def load_mff(path: str) -> mne.io.Raw:
    """
    Read mff EEG data into a Raw Object

    mne.io.Raw attributes:
        [obj].info(): overview info
        [obj].ch_names(): channel names
        [obj].n_times(): number of time points
        ... for more consult https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw
    """
    assert os.path.isdir(path), f"{path} does not exist"
    raw_mff = mne.io.read_raw_egi(path)
    return raw_mff

