#!/bin/bash

cd /home/zheng/DnCNN-PyTorch

python3 train.py \
  --num_of_layers 7 \
  --mode S \
  --noiseL 25 \
  --val_noiseL 25 \
  --outf results/logs_l7_1e-3 \
  --save_dir results/logs_l7_1e-3 \
  --lr 0.001 \
  --uncertainty	--gpu_ids 0
