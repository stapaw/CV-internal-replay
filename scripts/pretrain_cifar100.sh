#!/bin/bash
CUDA_VISIBLE_DEVICES=1

FC_LAYERS=3
HIDDEN_NEURONS=2000
ITERS=1
EPOCHS=100

# Other batch-replay size
MODEL_TAG=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_${EPOCHS}epochs_128br_proper

./main_pretrain.py --experiment=CIFAR100 --epochs=20 --augment --pre-convE --convE-ltag=${MODEL_TAG} --freeze-convE --full-stag=cifar100_pretrained_3hidden  --fc-layers=${FC_LAYERS} --fc-units=${HIDDEN_NEURONS}