#!/bin/bash

for TEST in ImageNetV2 ImageNetSketch ImageNetA ImaeNetR
do
  for CFG in vit_b16 vit_b32
  do
    FEAT_DIR="${CFG}/clip_feat"

    python linear_probe.py \
    --trainval_dataset "ImageNet" \
    --test_dataset ${TEST} \
    --num_step 8 \
    --feature_dir ${FEAT_DIR} \
    --num_run 3
  done
done
