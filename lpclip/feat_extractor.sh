#!/bin/bash


DATA="E:/Datasets/VisualClassification"

SEED=1

for CFG in vit_b16 vit_b32
do
  OUTPUT="./${CFG}/clip_feat"
#  for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 caltech101 ucf101 sun397
  for DATASET in imagenet imagenetv2 imagenet_sketch imagenet_a imagenet_r
  do
    for SPLIT in train val test
    do
      python feat_extractor.py \
      --split ${SPLIT} \
      --root ${DATA} \
      --seed ${SEED} \
      --dataset_config_file ../configs/datasets/${DATASET}.yaml \
      --config_file ../configs/trainers/${CFG}.yaml \
      --output_dir ${OUTPUT} \
      --eval-only
    done
  done
done
