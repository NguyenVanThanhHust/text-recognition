# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
from os.path import join
import sys
from datetime import datetime
from os import mkdir

import torch
import torch.nn.functional as F
import torchmetrics

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model, build_losses
from solver import make_optimizer
from torchmetrics.classification import Dice, BinaryJaccardIndex

from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter

def train(cfg, logger, output_dir):
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE) 

    optimizer = make_optimizer(cfg, model)
    scheduler = None

    train_loader = make_data_loader(cfg, split="train")
    val_loader = make_data_loader(cfg, split="val")

    writer = SummaryWriter(output_dir)

    iou_metric = BinaryJaccardIndex(
        threshold=0.5, 
    )
    dice_metric = Dice(
        num_classes=1, 
        threshold=0.5
    )

    metrics = {
        "iou": iou_metric, 
        "dice": dice_metric
    }

    losses = build_losses(cfg)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        device,
        optimizer,
        scheduler,
        losses,
        metrics,
        logger, 
        writer,
        output_dir
    )

    writer.close()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config_filename = os.path.basename(args.config_file).split(".")[0]
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    output_dir = join(output_dir, config_filename + "_" + dt_string)
    train(cfg, logger, output_dir)

if __name__ == '__main__':
    main()
