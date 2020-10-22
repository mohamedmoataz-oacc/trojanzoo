# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor

from trojanzoo.model.image.magnet import MagNet as MagNet_Model

import torch


class MagNet(Defense_Backdoor):
    name: str = 'magnet'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.magnet: MagNet_Model = MagNet_Model(dataset=self.dataset, pretrain=True)

    def detect(self, **kwargs):
        super().detect(**kwargs)
        self.validate_func()

    def get_data(self, data: (torch.Tensor, torch.LongTensor), org: bool = False, keep_org: bool = True, poison_label=True, **kwargs) -> (torch.Tensor, torch.LongTensor):
        if org:
            _input, _label = self.model.get_data(data)
        else:
            _input, _label = self.attack.get_data(data=data, keep_org=keep_org, poison_label=poison_label, **kwargs)
        _input = self.magnet(_input)
        return _input, _label

    def validate_func(self, **kwargs) -> (float, float, float):
        clean_loss, clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                                        get_data=self.get_data, org=True, **kwargs)
        target_loss, target_acc, _ = self.model._validate(print_prefix='Validate Trigger Tgt',
                                                          get_data=self.get_data, keep_org=False, **kwargs)
        _, orginal_acc, _ = self.model._validate(print_prefix='Validate Trigger Org',
                                                 get_data=self.get_data, keep_org=False, poison_label=False, **kwargs)
        print(f'Validate Confidence : {self.attack.validate_confidence():.3f}')
        return clean_loss + target_loss, target_acc, clean_acc
