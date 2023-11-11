# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import BinarizeMask


def build_transforms(cfg, split="train"):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if split=="train":
        img_transform = T.Compose([
            T.Resize(size=cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
        ])
        mask_transform = T.Compose([
            T.Resize(size=cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            BinarizeMask()
        ])
    else:
        img_transform = T.Compose([
            T.Resize(size=cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform,
        ])
        mask_transform = T.Compose([
            T.Resize(size=cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            BinarizeMask()
        ])
    return img_transform, mask_transform
