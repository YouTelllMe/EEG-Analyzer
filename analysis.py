import mne 
import numpy as np

def select_frequencies(data: mne.io.Raw | mne.Epochs,
                       freq: list[int]
                       ) -> tuple[list[list[list[float]]],
                                  list[float]]:
    """
    spec_data: epoch[channel[freq]]
    """
    data_spectrum = data.copy().compute_psd(method="welch")
    spec_data, spec_freqs = data_spectrum.get_data(return_freqs=True)
    spec_data = np.array(spec_data)
    spec_freqs = np.array(spec_freqs)

    freq_ind = 0
    selected_index = []

    for spec_ind in range(len(spec_freqs)):
        while True:
            if (freq_ind >= len(freq)):
                break
            elif abs(spec_freqs[spec_ind]-freq[freq_ind])<0.5:
                selected_index.append(spec_ind)
                break
            elif spec_freqs[spec_ind]<freq[freq_ind]:
                break
            else:
                freq_ind += 1

    selected_freq = np.take(spec_freqs, selected_index)
    selected_data = np.take(spec_data, selected_index, axis=2)
    
    return (selected_freq, selected_data)


def inverse_fourier(data: np.ndarray):
    """
    """
    return np.fft.ifft(data)
