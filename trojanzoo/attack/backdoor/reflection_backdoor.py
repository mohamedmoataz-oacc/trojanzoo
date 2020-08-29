from trojanzoo.attack import Attack
from trojanzoo.utils.mark import Watermark
from trojanzoo.utils import save_tensor_as_img

from typing import Union, List
from .badnet import BadNet
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import numpy as np

class Reflection_Backdoor(BadNet):
    name: str = 'reflection_backdoor'

    def __init__(self, reflect_num: int=20, selection_step: int=50, poison_num: int=1000,
                epoch: int=50, **kwargs):
        super().__init__(**kwargs)

        self.reflect_num: int = reflect_num
        self.selection_step: int = selection_step
        self.m: int = self.reflect_num//2
        self.poison_num: int = poison_num
        self.epoch: int = epoch

        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], device='cpu')
        self.conv2d = nn.Conv2d(1, 1, 3, bias=False)
        self.conv2d.weight = nn.Parameter(kernel.view_as(self.conv2d.weight))

        loader = self.dataset.get_dataloader(mode='train', batch_size=self.reflect_num, classes=[self.target_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
        self.reflect_set, self.reflect_labels = next(iter(loader)) # _images, _labels = next(iter(loader))
        self.W = torch.ones(reflect_num)

        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        self.train_loader = self.dataset.get_dataloader(mode='train', batch_size=self.poison_num, classes=other_classes,
                                                        shuffle=True, num_workers=0, pin_memory=False)
        self.valid_loader = self.dataset.get_dataloader(mode='validate',batch_size=self.poison_num, classes=other_classes,
                                                        shuffle=True, num_workers=0, pin_memory=False)
    
    def attack(self, save=False, **kwargs):
        # indices
        pick_img_ind = np.random.choice(len(range(self.reflect_num)), self.m, replace=False).tolist()
        ref_images = self.reflect_set[pick_img_ind]
        ref_labels = self.reflect_labels[pick_img_ind]

        for _ in range(self.selection_step):
            posion_imgs_train, labels_train = next(iter(self.train_loader))
            posion_imgs_valid, labels_valid = next(iter(self.valid_loader))
            for i in range(len(ref_images)):
                # locally change
                self.mark.mark = self.conv2d(ref_images[i])
                _posion_imgs_train = self.mark.add_mark(posion_imgs_train)
                _poison_imgs_valid = self.mark.add_mark(posion_imgs_valid)
                # todo
                self.model._train(self.epoch, save=save, validate_func=self.validate_func, 
                                  get_data=get_data, save_fn=self.save, **kwargs)
                # todo
                _, attack_acc, _ = self.validate_func(print_prefix='',
                                               get_data=None, **kwargs)
                self.W[pick_img_ind[i]] = attack_acc.item() 
                # todo: restore model

            # update self.W
            other_img_ind = list(set(range(self.reflect_num)) - set(pick_img_ind))
            self.W[other_img_ind] = self.W.median()

            # re-pick top m reflection images
            pick_img_ind = torch.argsort(self.W).tolist()[:self.m]
            ref_images = self.reflect_set[pick_img_ind]

        best_mark_ind = torch.argsort(self.W).tolist()[0]
        self.mark.mark = self.conv2d(ref_images[i])
        posion_imgs_train = self.mark.add_mark(posion_imgs_train)
        poison_imgs_valid = self.mark.add_mark(posion_imgs_valid)

        # todo
        self.model._train(epoch, save=save, validate_func=self.validate_func, 
                            get_data=get_data, save_fn=self.save, **kwargs)