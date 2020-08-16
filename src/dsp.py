from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sp

import pycwt as wavelet
from pycwt.helpers import find


def wavelet_decompose_power_spectrum(signal, title='', label='', units='', resample=None):
    if resample is not None:
        signal = sp.resample(signal, signal.shape[0] // resample)
    t = np.arange(signal.shape[0])
    N = t.shape[0]
    dt = 1

    p = np.polyfit(t, signal, 1)
    dat_notrend = signal - np.polyval(p, t)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset

    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves
    alpha, _, _ = wavelet.ar1(signal)  # Lag-1 autocorrelation for red noise

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    power /= scales[:, None]

    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # Prepare the figure
    plt.close('all')
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # First sub-plot, the original time series anomaly and inverse wavelet
    # transform.
    ax1.plot(t, signal, 'g', linewidth=1, label='Original signal')
    ax1.plot(t, iwave, '-r', linewidth=1, label='Inverse wavelet')
    ax1.legend()
    ax1.set_title('a) {}'.format(title))
    ax1.set_ylabel(r'{} [{}]'.format(label, units))

    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    # print(period)
    # print(power)
    # print(levels)
    ax2.contourf(t, np.log2(freqs), np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.gray)
    extent = [t.min(), t.max(), 0, max(period)]
    # ax2.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
    #                            t[:1] - dt, t[:1] - dt]),
    #         np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
    #                            np.log2(period[-1:]), [1e-9]]),
    #         'k', alpha=0.3, hatch='x')
    ax2.set_title('b) {} Wavelet Power Spectrum ({})'.format(label, mother.name))
    ax2.set_ylabel('Period (years)')
    #
    Yticks = 2 ** np.arange(np.ceil(np.log2(freqs.min())),
                               np.ceil(np.log2(freqs.max())))
    ax2.set_yticks(np.log2(Yticks))
    ax2.set_yticklabels(Yticks)

    plt.show()
