import os
import time

import torch
import wfdb
import dsp
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pycwt import Morlet
from tqdm import tqdm
from wfdb import processing as wfdb_processing
from collections import Counter


def read_record(data_path, header, offset=0, sample_size_seconds=30, samples_per_second=250):
    # Sample configuration
    sample_size = sample_size_seconds * samples_per_second

    max_sampto = header.sig_len
    sampto = offset + sample_size
    if sampto <= offset or sampto > max_sampto:
        return None, None, None
    record = wfdb.rdrecord(data_path + header.record_name, sampfrom=offset, sampto=sampto)

    ann = wfdb.rdann(data_path + header.record_name, 'atr', sampfrom=offset, sampto=sampto)
    return record, ann, sampto


def read_records(dataset_name, data_path, sample_size_seconds=30, samples_per_second=250, num_records=None):
    samples = []
    labels = []
    total_read_records = 0
    for record_name in wfdb.get_record_list(dataset_name):
        header = wfdb.rdheader(data_path + record_name)

        if header.sig_len == 0:
            continue

        offset = 0
        samples_count = 0
        while True:
            record, ann, offset = read_record(data_path, header, offset, sample_size_seconds, samples_per_second)
            if record is None:
                break
            samples.append(record)
            labels.append(ann.aux_note)
            samples_count += 1

        total_read_records += 1
        if num_records is not None and total_read_records == num_records:
            break

    # for l in labels:
    #     print(Counter(l))
    labels = np.array([1 if '(AFIB' in key or '(N' in key or '(AFL' in key or '(J' in key else 0 for key in labels])
    # print(labels)
    return samples, labels


def split_sample(record: wfdb.Record, sample_size_seconds=30, samples_per_second=250, device=None):
    sample: torch.Tensor = torch.tensor(record.p_signal) \
        .to(device) \
        .transpose(0, 1)[0]
    chunk_size = sample_size_seconds * samples_per_second
    return sample.view(-1, chunk_size)


class AFECGDataset(Dataset):
    """Atrial Fibrilation ECG dataset"""

    def __init__(self, dataset_name, data_path, samples_per_second=250, seq_len=20, wavelet=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.samples_per_second = samples_per_second
        self.seq_len = seq_len
        self.samples = []
        self.labels = []
        self.to_wavelet = wavelet
        # self.transform_samples = [split_sample(sample) for sample in self.samples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if type(index) == int and index < 0 or index > len(self.samples):
            return None
        # samples_per_interval = split_sample(self.samples[index])
        return self.samples[index], self.labels[index]

    def load(self, backup_path=None, num_records=None, checkpoint_interval=100):
        samples_per_second = self.samples_per_second
        to_wavelet = self.to_wavelet

        if backup_path is not None:
            filename = ('{}.pt' if self.to_wavelet is None else '{}_transformed.pt').format(self.dataset_name)
            file_path = os.path.join(backup_path, filename)
            if not os.path.exists(backup_path):
                os.makedirs(backup_path)
        else:
            file_path = None

        if file_path is not None and os.path.isfile(file_path):
            data = torch.load(file_path)
            print('Loaded from backup')
            self.samples, self.labels = data['samples'], data['labels']
            return

        data, labels = read_records(self.dataset_name, self.data_path,
                                    sample_size_seconds=samples_per_second * self.seq_len,
                                    samples_per_second=samples_per_second, num_records=num_records)
        labels = torch.tensor(labels)
        data = [split_sample(sample) for sample in data]
        count = len(data)
        transformed_data = []

        print('Preparing {} samples'.format(count))

        num_samples = 0
        for sample_idx, sample in tqdm(enumerate(data[:count]), desc='Preprocessing examples'):
            wavelets = []

            for signal_idx, signal in enumerate(sample):
                signal = wfdb_processing.normalize_bound(signal.numpy())
                if to_wavelet is not None:
                    sw = to_wavelet(signal)
                else:
                    sw = signal
                wavelets.append(torch.tensor(sw))

            t = torch.stack(wavelets)
            transformed_data.append(t)
            num_samples += 1

        transformed_data = torch.stack(transformed_data)
        if backup_path is not None:
            torch.save({
                'samples': transformed_data,
                'labels': labels
            }, file_path)

        self.samples, self.labels = transformed_data, labels


def _select_best_signal_fit(signals):
    return signals[0]  # TODO Select by SQI


class SecondDataset(Dataset):
    """
    This dataset loads ECG samples from the AFDB dataset one by one, each containing a single label.
    it is possible to control the size of each window (in seconds) and load data from other sources by tuning the sample
    rate (samples_per_second)
    """
    def __init__(self, dataset_name, data_path, samples_per_second=250, sample_size_seconds=30, wt=None,
                 signal_selector=_select_best_signal_fit, normalize=True):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.samples_per_second = samples_per_second
        self.sample_size_seconds = sample_size_seconds
        self.to_wavelet = wt
        self.samples = torch.tensor([])  # Compatibility with __len__
        self.labels = torch.tensor([])
        self.signal_selector = signal_selector
        self.normalize = normalize

    def load(self, backup_path=None):
        filename = ('dataset.pt' if self.to_wavelet is None else 'dataset_transformed.pt').format(self.dataset_name)
        if backup_path is not None and os.path.exists(os.path.join(backup_path, filename)):
            dataset_path = os.path.join(backup_path, filename)
            dataset_loaded = torch.load(dataset_path)
            self.samples = dataset_loaded['samples']
            self.labels = dataset_loaded['labels']
            print('Loaded {} samples from backup'.format(self.samples.shape[0]))
        else:
            samples_per_second = self.samples_per_second
            sample_size_seconds = self.sample_size_seconds
            records, labels = read_records(self.dataset_name, self.data_path, sample_size_seconds=sample_size_seconds,
                                           samples_per_second=samples_per_second)
            self.labels = torch.tensor(labels)
            tensors = []
            for record in tqdm(records):
                sample: torch.Tensor = self.signal_selector(torch.tensor(record.p_signal).transpose(0, 1)) \
                    if self.signal_selector is not None else torch.tensor(record.p_signal).transpose(0, 1)
                if self.to_wavelet is not None:
                    sw = self.to_wavelet(sample)
                else:
                    sw = sample

                if self.normalize:
                    sw = (sw - sw.min()) / (sw.max() if sw.max() != 0 else 1)
                tensors.append(sw)
            self.samples = torch.stack(tensors)
            if backup_path is not None:
                dataset_path = os.path.join(backup_path, filename)
                if not os.path.exists(backup_path):
                    os.makedirs(backup_path)
                torch.save({
                    'samples': self.samples,
                    'labels': self.labels
                }, dataset_path)

        print(self.labels.shape)
        print(self.samples.shape)
        assert isinstance(self.labels, torch.Tensor)
        assert isinstance(self.samples, torch.Tensor)
        assert self.labels.shape[0] == self.samples.shape[0]

    def __getitem__(self, index):
        if type(index) == int and index < 0 or index > len(self.samples):
            return None
        return self.samples[index], self.labels[index]

    def __len__(self) -> int:
        return self.samples.shape[0]


class WrapperDataset(Dataset):
    def __init__(self, samples: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index: int):
        if type(index) == int and index < 0 or index > self.samples.shape[0]:
            return None
        return self.samples[index], self.labels[index]

    def __len__(self) -> int:
        return self.samples.shape[0]


class WaveletTransform(object):
    """A Transform which enables a raw ECG signal to be transformed into a wavelet power spectrum, represented as
    a PyTorch Tensor object"""

    def __init__(self, wavelet=None, resample=None, resample_freq=None) -> None:
        super().__init__()
        self.wavelet = wavelet
        self.resample = resample
        self.resample_freq = resample_freq

    def __call__(self, sample):
        """
        Transform a single ECG signal into a 3-channel tensor image of a wavelet power spectrogram.
        :param sample: The sample, an array of size (N,) where N is the window size (in samples)
        :return: A Tensor of size (H, W) where WxH is the image size and each point is the power value of the specturm
        at the given time a frequence (after downsampling)
        """
        signal = sample
        time, frequencies, power, new_signal = dsp.wavelet_decompose_power_spectrum(signal, wl=self.wavelet,
                                                                                    resample=self.resample, resample_freq=self.resample_freq)
        # np_image = dsp.wavelet_figure_to_numpy_image(time, signal, frequencies, power, self.size[0], self.size[1], self.dpi, levels=levels)
        # t = torch.from_numpy(np_image)
        plt.close()
        t = torch.tensor(power)
        return t

    def to_file(self, path):
        # TODO Implement
        pass


def balanced_dataset(dataset, heldout_rate=0.2):
    heldout = int(len(dataset) * heldout_rate)
    train_size = len(dataset) - heldout
    true_examples = dataset.samples[dataset.labels == 1]
    true_labels = dataset.labels[dataset.labels == 1]
    false_examples = dataset.samples[dataset.labels == 0]
    false_labels = dataset.labels[dataset.labels == 0]

    true_train_size = int(len(true_examples) * (1 - heldout_rate))
    false_train_size = int(len(false_examples) * (1 - heldout_rate))

    true_examples_train, true_examples_test = true_examples[:true_train_size], true_examples[true_train_size:]
    true_labels_train, true_labels_test = true_labels[:true_train_size], true_labels[true_train_size:]

    false_examples_train, false_examples_test = false_examples[:false_train_size], false_examples[false_train_size:]
    false_labels_train, false_labels_test = false_labels[:false_train_size], false_labels[false_train_size:]

    augmented_examples_train = []
    augmented_labels_train = []
    for i in range(len(false_examples_train) // len(true_examples_train)):
        augmented_examples_train.append(true_examples_train.clone())
        augmented_labels_train.append(torch.ones(len(true_examples_train)).long())

    dataset.samples = torch.cat([torch.cat(augmented_examples_train), false_examples_train])
    dataset.labels = torch.cat([torch.cat(augmented_labels_train), false_labels_train])

    test_data = torch.cat([true_examples_test, false_examples_test])
    test_labels = torch.cat([true_labels_test, false_labels_test])

    train_dataset = dataset
    test_dataset = WrapperDataset(test_data, test_labels)
    return train_dataset, test_dataset