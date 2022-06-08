#!/bin/bash

FC_LAYERS=3
HIDDEN_NEURONS=2000
ITERS=5000
EPOCHS=100

# Other batch-replay size
BATCH_REPLAY=256
MODEL_TAG=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_${EPOCHS}epochs_${BATCH_REPLAY}br_freeze_classifier

./main_pretrain.py \
  --experiment=CIFAR10 \
  --epochs=${EPOCHS} \
  --augment \
  --fc-units=${HIDDEN_NEURONS} \
  --fc-layers=${FC_LAYERS} \
  --convE-stag=${MODEL_TAG} \


FC_LATENT_LAYER=0
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
  --fc-latent-layer=${FC_LATENT_LAYER}

FC_LATENT_LAYER=1
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
  --fc-latent-layer=${FC_LATENT_LAYER}

FC_LATENT_LAYER=2
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
  --fc-latent-layer=${FC_LATENT_LAYER}
#
#
#FC_LATENT_LAYER=3
#./main_cl.py \
#  --experiment=CIFAR100 \
#  --scenario=class \
#  --tasks=10 \
#  --batch=256 \
#  --batch-replay=${BATCH_REPLAY} \
#  --seed=0 \
#  --replay=generative \
#  --prior=GMM \
#  --per-class \
#  --g-z-dim=100 \
#  --sample-n=30 \
#  --pdf \
#  --depth=5 \
#  --fc-layers=${FC_LAYERS} \
#  --fc-units=${HIDDEN_NEURONS} \
#  --iters=${ITERS} \
#  --distill \
#  --freeze-convE \
#  --pre-convE \
#  --convE-ltag=${MODEL_TAG} \
#  --hidden \
#  --fc-latent-layer=${FC_LATENT_LAYER}