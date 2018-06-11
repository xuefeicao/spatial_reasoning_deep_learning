#!/bin/sh
GPU="3"
CUDA_VISIBLE_DEVICES=$GPU python train_vgg16.py
CUDA_VISIBLE_DEVICES=$GPU python test_vgg16.py