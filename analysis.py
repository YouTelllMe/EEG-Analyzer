import mne
import os


def load_mff(path: str) -> mne.io.Raw:
    """
    Read mff EEG data into a Raw Object
    """
    assert os.path.isdir(path), f"{path} does not exist"
    raw_mff = mne.io.read_raw_egi(path)
    return raw_mff