#!/bin/bash

# custom config
DATA="E:/Datasets/VisualClassification"
TRAINER="ZeroshotCLIP"
CFG="vit_b16" # rn50_val, b16_val

#for CFG in vit_b16
#do
#  for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 caltech101 ucf101 imagenet sun397
for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 caltech101 ucf101 imagenet sun397
do
  python run_zsclip.py \
  --dataset_root ${DATA} \
  --trainer ${TRAINER} \
  --dataset_config "../configs/datasets/${DATASET}.yaml" \
  --trainer_config "../configs/trainers/${TRAINER}/${CFG}.yaml" \
  --output_dir output/new/${CFG}/${DATASET} \
  --evaluator "Microf1Classification" \
  --eval_only \
  DATASET.SUBSAMPLE_CLASSES  "new"
done
#done
