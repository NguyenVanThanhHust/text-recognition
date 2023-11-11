import os, sys
from os.path import join

import torch
from torch.utils.data import Dataset

from PIL import Image

class CrackDataset(Dataset):
    def __init__(self, data_folder, img_transform=None, mask_transform=None) -> None:
        super().__init__()
        image_folder = join(data_folder, "images")
        mask_folder = join(data_folder, "masks")
        img_names = next(os.walk(image_folder))[2]
        self.img_paths = []
        self.mask_paths = []
        for img_name in img_names:
            img_path = join(image_folder, img_name)
            mask_path = join(mask_folder, img_name)
            self.img_paths.append(img_path)
            self.mask_paths.append(mask_path)

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return 2
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        
        mask = Image.open(mask_path)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        return img, mask

if __name__ == "__main__":
    crackDataset = CrackDataset("../Datasets/crack_segmentation_dataset/train")    
    print(crackDataset.__len__())
    for i in range(4):
        img, mask = crackDataset.__getitem__(i)
        print(img.size, mask.size)
        print(img.getextrema())
        print(mask.getextrema())