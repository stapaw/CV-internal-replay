#!/bin/bash

./main_cl.py \
  --experiment=fmnist \
  --scenario=class \
  --tasks 5 \
  --batch 128 \
  --seed 0 \
  --replay=generative \
  --prior=GMM \
  --per-class \
  --g-z-dim 10 \
  --sample-n 30 \
  --pdf \
  --depth=3 \
  --reducing-layers=2 \
  --channels=16 \
  --fc-layers=2 \
  --fc-units 100 \
  --gen-iters=2000 \
  --conv-bn=False \
  --distill \
  --freeze-convE \
  --freeze-fcE \
  --freeze-fcE-layer=2

./main_cl.py \
  --experiment=fmnist \
  --scenario=class \
  --tasks 5 \
  --batch 128 \
  --seed 0 \
  --replay=generative \
  --prior=GMM \
  --per-class \
  --g-z-dim 10 \
  --sample-n 30 \
  --pdf \
  --depth=3 \
  --reducing-layers=2 \
  --channels=16 \
  --fc-layers=2 \
  --fc-units 100 \
  --gen-iters=2000 \
  --conv-bn=False \
  --distill \
  --freeze-convE \
  --freeze-fcE \
  --freeze-fcE-layer=1

./main_cl.py \
  --experiment=fmnist \
  --scenario=class \
  --tasks 5 \
  --batch 128 \
  --seed 0 \
  --replay=generative \
  --prior=GMM \
  --per-class \
  --g-z-dim 10 \
  --sample-n 30 \
  --pdf \
  --depth=3 \
  --reducing-layers=2 \
  --channels=16 \
  --fc-layers=2 \
  --fc-units 100 \
  --gen-iters=2000 \
  --conv-bn=False \
  --distill \
  --freeze-convE

