import wfdb
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

    return samples, labels


def split_sample(record, sample_size_seconds=30, samples_per_second=250):
    sample = record.p_signal
    samples = np.split(sample, sample_size_seconds*samples_per_second)
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

    def __init__(self) -> None:
        super().__init__()
        # TODO Implement

    def __call__(self, sample):
        # TODO Implement
        pass
