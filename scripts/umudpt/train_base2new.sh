#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --no-requeue
#SBATCH --output=base2new.out

source activate py37_coop

PROJ_PATH="E:/work4_MuDPT/MuDPT"
DATA="E:/Datasets/VisualClassification"
TRAINER="UMuDPT"

CFG="vit_b16_bz4_ep5_nctx2_depth9"
BACKBONE_PATH="${PROJ_PATH}/pretrained/ViT-B-16.pt"
N_CTX=2
DEEP_PROMPT_DEPTH=9
SHOTS=16
SEED=2

for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 caltech101 ucf101 imagenet sun397
do
  DIR=output/base2new/base/${DATASET}/shots_${SHOTS}/${CFG}_nctx${N_CTX}_depth${DEEP_PROMPT_DEPTH}/seed${SEED}

  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
      python ${PROJ_PATH}/train.py \
      --dataset_root ${DATA} \
      --trainer ${TRAINER} \
      --seed ${SEED} \
      --dataset_config "${PROJ_PATH}/configs/datasets/${DATASET}.yaml" \
      --trainer_config "${PROJ_PATH}/configs/trainers/${TRAINER}/${CFG}.yaml" \
      --output_dir ${DIR} \
      DATASET.NUM_SHOTS ${SHOTS} \
      MODEL.BACKBONE.PATH ${BACKBONE_PATH} \
      DATASET.SUBSAMPLE_CLASSES "base"
  fi
done