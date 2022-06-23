#!/bin/bash
CUDA_VISIBLE_DEVICES=1

FC_LAYERS=4
HIDDEN_NEURONS=1000
ITERS=5000
EPOCHS=100
FREQUENCY=$2
SEED=$3

# Other batch-replay size
BATCH_REPLAY=$1
echo batch replay ${BATCH_REPLAY}
MODEL_TAG=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_${EPOCHS}epochs_128br_proper
RESULTS_DIR=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_${EPOCHS}epochs_${BATCH_REPLAY}br_${ITERS}iters_${FREQUENCY}frequency_fid


./main_cl.py \
  --experiment=CIFAR100 \
  --scenario=class \
  --tasks=10 \
  --batch=256 \
  --batch-replay=${BATCH_REPLAY} \
  --seed=${SEED} \
  --replay=generative \
  --prior=GMM \
  --per-class \
  --g-z-dim=100 \
  --sample-n=30 \
  --pdf \
  --depth=5 \
  --fc-layers=${FC_LAYERS} \
  --fc-units=${HIDDEN_NEURONS} \
  --iters=${ITERS} \
  --distill \
  --freeze-convE \
  --pre-convE \
  --convE-ltag=${MODEL_TAG} \
  --hidden \
  --res-dir=${RESULTS_DIR} \
  --latent-replay-layer-frequency=${FREQUENCY} \
  --latent \
  --eval-tag=cifar100_pretrained \
  --test

