#!/usr/bin/env python3
from .imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models
from collections import OrderedDict

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}

cifar10_inverted_residual_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 1],    # Stride 2 -> 1 for CIFAR-10
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class _MobileNet(_ImageModel):

    def __init__(self, sub_type: str = 'v3_small', **kwargs):
        super().__init__(**kwargs)
        if 'v2' in sub_type:
            inverted_residual_setting = cifar10_inverted_residual_setting if 'comp' in sub_type else None
            _model = torchvision.models.mobilenet.mobilenet_v2(num_classes=self.num_classes,
                                                               inverted_residual_setting=inverted_residual_setting)
            if 'comp' in sub_type:
                conv1: nn.Conv2d = _model.features[0][0]
                _model.features[0][0] = nn.Conv2d(conv1.in_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                                                  stride=1, padding=conv1.padding, dilation=conv1.dilation,
                                                  groups=conv1.groups, bias=False)
        elif sub_type == 'v3_large':
            _model = torchvision.models.mobilenet.mobilenet_v3_large(num_classes=self.num_classes)
        else:
            assert sub_type == 'v3_small'
            _model = torchvision.models.mobilenet.mobilenet_v3_small(num_classes=self.num_classes)
        self.features = _model.features
        self.classifier = _model.classifier


class MobileNet(ImageModel):

    def __init__(self, name: str = 'mobilenet', model: type[_MobileNet] = _MobileNet, **kwargs):
        try:
            sub_type: str = name[10:]
            assert sub_type in ['v2', 'v2_comp', 'v3_small', 'v3_large'], f'{name=}'
        except:
            raise AssertionError("model name should be in ['mobilenet_v2', 'mobilenet_v2_comp', "
                                 "'mobilenet_v3_small', 'mobilenet_v3_large']")
        super().__init__(name=name, model=model, sub_type=sub_type, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls[self.name]
        print('get official model weights from: ', url)
        return model_zoo.load_url(url, **kwargs)
