#!/bin/bash

cd /home/zheng/DnCNN-PyTorch

python3 train_unet.py \
  --mode S \
  --noiseL 25 \
  --val_noiseL 25 \
  --save_dir results/logs_unet  \
  --batchSize 16  \
  --epochs 100 \
  --datapath_train /mnt/ddisk/zzheng/dataset  \
  --gpu_ids 1
