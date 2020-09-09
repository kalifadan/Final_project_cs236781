from __future__ import division

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sp
import pycwt as wavelet


def plot_wavelet_decomposition(time, signal, frequencies, power, wavelet_name):

    # Prepare the figure
    plt.close('all')
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    # First sub-plot, the original time series anomaly.
    ax1.plot(time, signal, 'g', linewidth=1, label='Original signal')
    ax1.legend()
    ax1.set_title('a) Original ECG Signal')
    ax1.set_ylabel(r'Voltage [mV]')

    # Second subplot, the wavelet power spectrum
    ax2.contourf(time, frequencies, np.log2(power), extend='both', cmap=plt.cm.gray)
    ax2.set_title('b) Wavelet Power Spectrum ({})'.format(wavelet_name))
    ax2.set_ylabel('Frequency [Hz]')
    plt.show()


def wavelet_figure_to_numpy_image(time, signal, frequencies, power, width, height, dpi, levels=None):
    if levels is None:
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]

    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax.axis('off')
    ax.contourf(time, np.log2(frequencies), np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.gray)
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((width, height, -1))
    return image


def wavelet_decompose_power_spectrum(signal, wl=None,
                                     resample=None,
                                     resample_freq=None,
                                     sampling_frequency=None,
                                     filter_frequency=40,
                                     dt=1):
    """
    :param signal: The signal, a numpy array or PyTorch Tensor of shape (N,)
    :param wl: Provided Wavelet (see pycwt documentation for available wavelets)
    :param resample: Downsample factor for signal time series.
    :param resample_freq: Downsample factor for wavelet frequency plane.
    :param sampling_frequency: Sampling frequency to be used by the butterworth filter, if provided.
    :param filter_frequency: Filter frequency for the butterworth filter
    :param dt: Sampling interval Sampling interval for the continuous wavelet transform.
    :return: Resampled time series, Resamples frequency series, power spectrum of shape (Frequencies, Time),
    Original signal.
    """
    if resample is not None:
        signal = sp.resample(signal, signal.shape[0] // resample)

    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    # Butterworth filter
    if sampling_frequency is not None:
        sos = sp.butter(5, filter_frequency, 'low', fs=sampling_frequency, output='sos')
        signal = sp.sosfilt(sos, signal)

    time = np.arange(signal.shape[0])

    # p = np.polyfit(time, signal, 1)
    # dat_notrend = signal - np.polyval(p, time)
    # std = dat_notrend.std()  # Standard deviation
    # dat_norm = dat_notrend / std  # Normalized dataset

    if wl is None:
        wl = wavelet.Morlet(6)

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(signal, dt, wavelet=wl)
    power = (np.abs(wave)) ** 2

    power /= scales[:, None]

    if resample_freq is not None:
        power = sp.resample(power, num=resample_freq, axis=0)
        freqs = sp.resample(freqs, num=resample_freq)

    return time, np.array(freqs), power, signal
