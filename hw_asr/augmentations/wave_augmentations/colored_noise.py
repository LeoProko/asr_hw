import torch
import torch_audiomentations

from hw_asr.augmentations.base import AugmentationBase


class ColoredNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: torch.Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
