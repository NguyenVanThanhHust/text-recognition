# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch.utils import data

from .datasets.crack import CrackDataset
from .transforms import build_transforms


def build_dataset(data_folder, img_transforms, mask_transforms):
    datasets = CrackDataset(data_folder=data_folder, img_transform=img_transforms, mask_transform=mask_transforms)
    return datasets

def make_data_loader(cfg, split="train"):
    if split=="train":
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    split_datafolders = {
        "train": cfg.INPUT_TRAIN_FOLDER,
        "val": cfg.INPUT_VAL_FOLDER,
        "test": cfg.INPUT_TEST_FOLDER,
    }

    img_transforms, mask_transforms = build_transforms(cfg, split)
    datasets = build_dataset(split_datafolders[split], img_transforms, mask_transforms)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
