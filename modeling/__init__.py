# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch.nn.functional as F

from .unet import UNet
from .losses import Loss

def build_model(cfg):
    model = UNet(cfg.MODEL.NUM_CLASSES)
    return model

def build_losses(cfg):
    return Loss()