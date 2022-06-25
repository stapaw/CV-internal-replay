#!/bin/bash
CUDA_VISIBLE_DEVICES=1

FC_LAYERS=3
HIDDEN_NEURONS=2000
ITERS=5000
EPOCHS=100
BATCH_REPLAY=256
STRATEGY=$1
SEED=$2

# Other batch-replay size
BATCH_REPLAY=$1
echo batch replay ${BATCH_REPLAY}
MODEL_TAG=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_${EPOCHS}epochs_128br_proper
RESULTS_DIR=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_${EPOCHS}epochs_${BATCH_REPLAY}br_${ITERS}iters_${STRATEGY}strategy_fid


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
  --res-dir=${RESULTS_DIR} \
  --latent-replay-strategy=${STRATEGY} \
  --latent \
  --eval-tag=cifar100_pretrained_3hidden \
  --test

