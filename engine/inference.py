# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import math
import os
import sys
from typing import Iterable

import torch

@torch.no_grad()
def evaluate(model, data_loader, metrics, device):
    model.eval()
    model = model.to(device)
    print("Device: ", device)
    eval_metrics = dict()
    result_metrics = dict()
    for k, v in metrics.items():
        eval_metrics[k] = v.to(device)
        result_metrics[k] = list()

    for idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        iou_value = eval_metrics["iou"](outputs, targets.type(torch.int32))
        dice_value = eval_metrics["dice"](outputs.view(-1), targets.type(torch.int32).view(-1))
        
        result_metrics["iou"].append(iou_value)
        result_metrics["dice"].append(dice_value)

    epoch_eval_res = {}
    for k, v in eval_metrics.items():
        each_metric_res = v.compute()
        epoch_eval_res[k] = each_metric_res
    return epoch_eval_res