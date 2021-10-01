from functools import partial
import torch
from torch import nn

class NormalizationWrapper(nn.Module):
    def __init__(self, model, mean, std, channels_last=False, input_size=None, interpolation=None):
        super().__init__()
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        if not channels_last:
            mean = mean.view(-1,1,1)
            std = std.view(-1,1,1)

        if input_size is not None:
            size = input_size[:2] if channels_last else input_size[1:]
            self.upsample = nn.Upsample(input_size[1:], mode=interpolation, align_corners=True)
        else:
            self.upsample = nn.Identity()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.model = model

    def forward(self, x):
        x = self.upsample(x)
        return self.model((x - self.mean) / self.std)
