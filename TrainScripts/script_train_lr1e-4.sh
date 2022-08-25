#!/bin/bash

cd /home/zheng/DnCNN-PyTorch

python3 train.py \
  --num_of_layers 7 \
  --mode S \
  --noiseL 25 \
  --val_noiseL 25 \
  --outf results/logs_l7_1e-4 \
  --save_dir results/logs_l7_1e-4 \
  --lr 0.0001 \
  --uncertainty	--gpu_ids 1
