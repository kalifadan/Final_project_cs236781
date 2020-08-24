from __future__ import division
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sp

import pycwt as wavelet
from pycwt.helpers import find


def plot_wavelet_decomposition(time, signal, frequencies, power, wavelet_name, levels=None):
    if levels is None:
        levels = [-0.125, -0.0625, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]

    period = 1 / frequencies
    # Prepare the figure
    plt.close('all')
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # First sub-plot, the original time series anomaly and inverse wavelet
    # transform.
    ax1.plot(time, signal, 'g', linewidth=1, label='Original signal')
    # ax1.plot(time, iwave, '-r', linewidth=1, label='Inverse wavelet')
    ax1.legend()
    ax1.set_title('a) Original ECG Signal')
    ax1.set_ylabel(r'Frequency [Hz]')

    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    # levels = np.logspace(-10, 10, num=20, base=2)

    print('Time: ', time.shape)
    print('Frequencies: ', frequencies.shape)
    print('Power: ', power.shape)
    ax2.contourf(time, period, np.log2(power), extend='both', cmap=plt.cm.gray)
    extent = [time.min(), time.max(), 0, max(period)]
    # ax2.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
    #                            t[:1] - dt, t[:1] - dt]),
    #         np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
    #                            np.log2(period[-1:]), [1e-9]]),
    #         'k', alpha=0.3, hatch='x')
    ax2.set_title('b) Wavelet Power Spectrum ({})'.format(wavelet_name))
    ax2.set_ylabel('Frequency [Hz]')
    #
    # y_ticks = np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    # print(y_ticks)
    # ax2.set_yticks(np.log2(y_ticks))
    # ax2.set_yticklabels(y_ticks)
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
                                     significance_level=0.9,
                                     resample=None):
    if resample is not None:
        signal = sp.resample(signal, signal.shape[0] // resample)

    # Butterworth filter
    sos = sp.butter(5, 40, 'low', fs=1000, output='sos')
    signal = sp.sosfilt(sos, signal)

    time = np.arange(signal.shape[0])
    N = time.shape[0]
    dt = 1

    p = np.polyfit(time, signal, 1)
    dat_notrend = signal - np.polyval(p, time)
    std = dat_notrend.std()  # Standard deviation
    dat_norm = dat_notrend / std  # Normalized dataset

    if wl is None:
        wl = wavelet.Morlet(6)

    # TODO Check these hyperparams
    s0 = 8.33  # Starting scale
    dj = 1 / 15  # X sub-octaves per octaves
    J = 19  # Seven powers of two with dj sub-octaves
    # alpha, _, _ = wavelet.ar1(signal)  # Lag-1 autocorrelation for red noise

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, wl)
    # iwave = wavelet.icwt(wave, scales, dt, dj, wl) * std
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2

    power /= scales[:, None]

    # signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha, significance_level=significance_level, wavelet=wl)
    # sig_percentile = np.ones([1, N]) * signif[:, None]
    # sig_percentile = power / sig_percentile

    # period = 1 / freqs
    return time, np.array(freqs), power, signal
