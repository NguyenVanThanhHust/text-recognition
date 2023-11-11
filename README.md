# Crack Segmentation

# Table Of Contents
-  [How to run](#how-to-run)
-  [Acknowledgments](#acknowledgments)

# How to run
Download data from [crack_segmentation_repo](https://github.com/khanhha/crack_segmentation)
Split train set to train and validation set, modify `data_folder` accord to your dataset folder
```
python scripts/split_train_val.py --data_folder ../Datasets/ 
```   

How to train
```
python tools/train.py --config_file configs/simple_unet.yaml
```

How to infer
```
python tools/test --config_file configs/simple_unet.yaml TEST.WEIGHT your_trained_weight_here
```

# Acknowledgments
Unet Implementation: https://github.com/usuyama/pytorch-unet
Deep-Learning project template: 