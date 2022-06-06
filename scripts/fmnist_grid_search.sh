#!/bin/bash

FC_LAYERS=3
HIDDEN_NEURONS=2000
MODEL_TAG=cifar10_${FC_LAYERS}fc_${HIDDEN_NEURONS}hn_proper

BATCH_REPLAY=256
ITERS=5000

#./main_pretrain.py \
#  --experiment=CIFAR10 \
#  --epochs=100 \
#  --augment
#  --fc-units=${HIDDEN_NEURONS} \
#  --fc-layers=${FC_LAYERS} \
#  --convE-stag=${MODEL_TAG}

# Brain inspired sanity check
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
#  --brain-inspired \
#  --pre-convE \
#  --convE-ltag=${MODEL_TAG} \

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
#
#FC_LATENT_LAYER=2
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
#
## Other batch-replay size
#BATCH_REPLAY=128
#
#./main_pretrain.py \
#  --experiment=CIFAR10 \
#  --epochs=100 \
#  --batch=256 \
#  --depth=5 \
#  --fc-units=${HIDDEN_NEURONS} \
#  --fc-layers=${FC_LAYERS} \
#  --convE-stag=${MODEL_TAG}
#
#
#FC_LATENT_LAYER=0
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
#
#FC_LATENT_LAYER=1
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
#
#FC_LATENT_LAYER=2
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
#
## Other batch-replay size
#BATCH_REPLAY=512
#
#./main_pretrain.py \
#  --experiment=CIFAR10 \
#  --epochs=100 \
#  --batch=256 \
#  --depth=5 \
#  --fc-units=${HIDDEN_NEURONS} \
#  --fc-layers=${FC_LAYERS} \
#  --convE-stag=${MODEL_TAG}
#
#
#FC_LATENT_LAYER=0
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
#
#FC_LATENT_LAYER=1
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
#
#FC_LATENT_LAYER=2
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
