# EEG-Analyzer

Extension of the MNE Python package for further EEG analysis

## TO-DO

1. do filtering, epoch'ing and potentially artifact rejection
2. runs Fourier analysis and identifies harmonics in the spectrum based on known stimulation frequencies
3. aggregates the data and runs summary statistics using vector-based approaches that takes into account both the phase and amplitude of the response
4. uses an inverse Fourier transform to generate waveforms filtered to contain only certain harmonics
5. plots the data and outputs data for further analysis
