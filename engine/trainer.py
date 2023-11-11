# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import math
import os
from os.path import join
import sys
from typing import Iterable

import torch


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    scheduler, 
                    device: torch.device, epoch: int, logger, writer):
    model.train()
    model = model.to(device)

    for idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        loss_value = losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        logger.info("loss: {}".format(loss_value))
        writer.add_scalar("Loss/train", loss_value, epoch)

@torch.no_grad()
def evaluate(model, criterion, data_loader, metrics, device, epoch, logger, writer):
    model.eval()
    model = model.to(device)
    eval_metrics = dict()
    result_metrics = dict()
    for k, v in metrics.items():
        eval_metrics[k] = v.to(device)
        result_metrics[k] = list()

    for idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        iou_value = eval_metrics["iou"](outputs, targets.type(torch.int32))
        dice_value = eval_metrics["dice"](outputs.view(-1), targets.type(torch.int32).view(-1))
        
        result_metrics["iou"].append(iou_value)
        result_metrics["dice"].append(dice_value)
        loss_value = losses.item()
        logger.info("loss: {}".format(loss_value))
        writer.add_scalar("Loss/eval", loss_value)
        for k, v in result_metrics.items():
            writer.add_scalar(k, v[-1])

    epoch_eval_res = {}
    for k, v in eval_metrics.items():
        each_metric_res = v.compute()
        logger.info("Epoch: {} {} {}".format(epoch, k, each_metric_res))
        epoch_eval_res[k] = each_metric_res
    return epoch_eval_res


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        device, 
        optimizer,
        scheduler,
        loss_fn,
        metrics, 
        logger, 
        writer, 
        output_dir, 
):
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger = logging.getLogger("template_model.train")
    logger.info("Start training")

    best_result = {}
    for k, v in metrics.items():
        best_result[k] = 0.0

    for epoch in range(epochs):
        train_one_epoch(model, loss_fn, train_loader, optimizer, scheduler, device, epoch, logger, writer)
        epoch_eval_res = evaluate(model, loss_fn, val_loader, metrics, device, epoch, logger, writer)
        for k, v in epoch_eval_res.items():
            if v > best_result[k]:
                best_result[k] = v
                best_model_path = join(output_dir, 'best_ckpt_{}.pth'.format(k))
                logger.info("Save best model at epoch: {}".format(epoch))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, best_model_path)
        latest_model_path = join(output_dir, 'latest_ckpt.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, latest_model_path)