import os, sys, shutil
import random
from os.path import join 
import argparse

from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser("Crack Segmentation")
    parser.add_argument('--data_folder', type=str, help='data folder contains train, test folder')
    parser.add_argument('--train_val_ratio', default=0.8, type=float, help='train val split')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    data_folder = args.data_folder
    train_folder = join(data_folder, "train")
    val_folder = join(data_folder, "val")
    os.makedirs(val_folder, exist_ok=True)
    val_img_folder = join(val_folder, "images")
    val_mask_folder = join(val_folder, "masks")
    os.makedirs(val_img_folder, exist_ok=True)
    os.makedirs(val_mask_folder, exist_ok=True)

    image_folder = join(train_folder, "images")
    mask_folder = join(train_folder, "masks")
    image_names = next(os.walk(image_folder))[2]
    random.shuffle(image_names)
    num_train_images = int(len(image_names) * args.train_val_ratio)
    for i in tqdm(range(num_train_images, len(image_names))):
        img_name = image_names[i]
        old_img_path = join(image_folder, img_name)
        old_mask_path = join(mask_folder, img_name)
        new_img_path = join(val_img_folder, img_name)
        new_mask_path = join(val_mask_folder, img_name)
        shutil.move(old_img_path, new_img_path)
        shutil.move(old_mask_path, new_mask_path)

