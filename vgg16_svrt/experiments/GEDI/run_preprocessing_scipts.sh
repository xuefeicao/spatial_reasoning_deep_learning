#!/bin/sh

python extract_frames_from_GEDI.py
python prepare_tf_records.py
#CUDA_VISIBLE_DEVICES=3 python train_vgg16.py