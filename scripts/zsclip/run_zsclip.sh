#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=run_coop_cls.out

source activate py37_coop

# custom config
DATA=""
TRAINER="ZeroshotCLIP"
# CFG="b16_val" # rn50_val, b16_val

for CFG in vit_b16
do
#  for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 caltech101 ucf101 imagenet sun397
  for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r
  do
    python run_zsclip.py \
    --dataset_root ${DATA} \
    --trainer ${TRAINER} \
    --dataset_config "../configs/datasets/${DATASET}.yaml" \
    --trainer_config "../configs/trainers/${CFG}.yaml" \
    --output_dir output/${CFG}/${DATASET} \
    --evaluator "Microf1Classification" \
    --eval_only
  done
done
