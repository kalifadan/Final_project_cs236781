import PIL
import torch
from torchvision.transforms import ToPILImage


def show_spectrogram(spectrogram: torch.Tensor, resize=None):
    to_image = ToPILImage(mode='L')

    min_v = torch.min(spectrogram)
    range_v = torch.max(spectrogram) - min_v
    if range_v > 0:
        normalised = 255 * (spectrogram - min_v) / range_v
    else:
        normalised = torch.zeros(spectrogram.size())

    normalised = normalised.int()
    i = to_image(spectrogram.float())
    if resize is not None:
        i = i.resize((300, 300), resample=PIL.Image.BICUBIC)
    return i.show()
