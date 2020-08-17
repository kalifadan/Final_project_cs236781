import torch
import wfdb
import dsp
import numpy as np
from torch.utils.data import Dataset, DataLoader


def read_record(data_path, header, offset=0, sample_size_seconds=30, samples_per_second=250):
    # Sample configuration
    sample_size = sample_size_seconds * samples_per_second

    max_sampto = header.sig_len
    sampto = min(max_sampto, offset + sample_size)
    if sampto <= offset:
        return None, None, None
    record = wfdb.rdrecord(data_path + header.record_name, sampfrom=offset, sampto=sampto)
    ann = wfdb.rdann(data_path + header.record_name, 'atr', sampfrom=offset, sampto=sampto)
    return record, ann, sampto


def read_records(dataset_name, data_path, sample_size_seconds=30, samples_per_second=250, batch_size=100):
    samples = []
    labels = []
    # Sample configuration
    sample_size = sample_size_seconds * samples_per_second

    for record_name in wfdb.get_record_list(dataset_name):
        header = wfdb.rdheader(data_path + record_name)

        if header.sig_len == 0:
            continue

        offset = 0
        samples_count = 0
        while True:
            max_sampto = header.sig_len
            record, ann, offset = read_record(data_path, header, offset, sample_size_seconds, samples_per_second)
            if record is None:
                break
            samples.append(record)
            labels.append(ann.aux_note)
            samples_count += 1
            if batch_size is not None and samples_count == batch_size:
                break

    labels = np.array([1 if '(AFIB' in key else 0 for key in labels])
    return samples, labels


def split_sample(record, sample_size_seconds=30, samples_per_second=250):
    sample = record.p_signal
    num_chunks = sample.shape[0] // (sample_size_seconds*samples_per_second)
    samples = list(map(lambda x: x.T[0], np.split(sample, num_chunks)))
    return samples


class AFECGDataset(Dataset):
    """Artirial Fibrilation ECG dataset"""

    def __init__(self, dataset_name, data_path, samples_per_second=250, batch_size=None) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path



        self.samples, self.labels = read_records(self.dataset_name, self.data_path, sample_size_seconds=10 * 60,
                                                 samples_per_second=samples_per_second, batch_size=batch_size)

    def __getitem__(self, index: int):
        # TODO Implement
        if index < 0 or index > len(self.samples):
            return None
        samples_per_interval = split_sample(self.samples[index])
        return samples_per_interval, self.labels[index]


class WaveletTransform(object):
    """A Transform which enables a raw ECG signal to be transformed into a wavelet power spectrum, represented as
    a PyTorch Tensor object"""

    def __init__(self, wavelet=None, size=(800, 800)) -> None:
        super().__init__()
        self.wavelet = wavelet
        self.size = size
        self.dpi = 72

    def __call__(self, sample):
        """
        Transform a single ECG signal into a 3-channel tensor image of a wavelet power spectrogram.
        :param sample: The sample, an array of size (N,) where N is the window size (in samples)
        :return: A Tensor of size (W, H, C) where WxH is the image size and C is the number of color channels
        """
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        signal = sample
        time, frequencies, power = dsp.wavelet_decompose_power_spectrum(signal, wl=self.wavelet)
        print(time.shape)
        print(frequencies.shape)
        np_image = dsp.wavelet_figure_to_numpy_image(time, signal, frequencies, power, self.size[0], self.size[1], self.dpi, levels=levels)
        print(np_image.shape)
        t = torch.from_numpy(np_image)
        self._t = t
        return t

    def to_file(self, path):
        # TODO Implement
        pass
