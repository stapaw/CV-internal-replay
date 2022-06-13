#!/bin/bash
CUDA_VISIBLE_DEVICES=1

FC_LAYERS=$1
HIDDEN_NEURONS=$2
ITERS=$3
EPOCHS=$4
FREQUENCY=$5
BATCH_REPLAY=$6
FULL_LTAG=$7
SUFFIX=$8

echo batch replay ${BATCH_REPLAY}

MODEL_TAG=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_${EPOCHS}epochs_128br_proper
RESULTS_DIR=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_${EPOCHS}epochs_${BATCH_REPLAY}br_${ITERS}iters_${FREQUENCY}${SUFFIX}


./main_cl.py \
  --experiment=CIFAR100 \
  --scenario=class \
  --tasks=10 \
  --batch=256 \
  --batch-replay=${BATCH_REPLAY} \
  --seed=0 \
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
  --latent \
  --eval-tag=cifar100_pretrained \
  --full-ltag=${FULL_LTAG} \
  --latent-replay-layer-frequency=${FREQUENCY} \
  --test

