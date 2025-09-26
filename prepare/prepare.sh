#!/usr/bin/env bash

dataset="cifar10"

CUDA_VISIBLE_DEVICES=0 python label.py $dataset

CUDA_VISIBLE_DEVICES=0 python txt_annotation.py $dataset

CUDA_VISIBLE_DEVICES=0 python LE.py 100 $dataset

CUDA_VISIBLE_DEVICES=0 python BLIP_generate.py $dataset

CUDA_VISIBLE_DEVICES=0 python pool.py $dataset