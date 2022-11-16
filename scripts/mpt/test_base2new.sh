#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --no-requeue
#SBATCH --output=base2new_new.out

source activate py37_coop

PROJ_PATH="/data01/home/scy0037/run/myz/MuDPT"
DATA="${PROJ_PATH}/datasets"

TRAINER="MPT"
CFG="vit_b16_c2_ep5_batch4"
BACKBONE_PATH="${PROJ_PATH}/pretrained/ViT-B-16.pt"
TEXT_N_CTX=2
DEEP_TEXT_N_CTX=2
DEEP_IMG_N_CTX=2
TEXT_PROMPT_DEPTH=12
IMG_PROMPT_DEPTH=12
SHOTS=16
SEED=2

#
for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 caltech101 ucf101 imagenet sun397
do
  MODEL_DIR=output/base2new/base/${DATASET}/shots_${SHOTS}/${CFG}_vctx${DEEP_IMG_N_CTX}_tctx${DEEP_TEXT_N_CTX}_vdepth${IMG_PROMPT_DEPTH}_tdepth${TEXT_PROMPT_DEPTH}/seed${SEED}
  OUTPUT_DIR=output/base2new/new/${DATASET}/shots_${SHOTS}/${CFG}_vctx${DEEP_IMG_N_CTX}_tctx${DEEP_TEXT_N_CTX}_vdepth${IMG_PROMPT_DEPTH}_tdepth${TEXT_PROMPT_DEPTH}/seed${SEED}
  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
      python ${PROJ_PATH}/train.py \
      --dataset_root ${DATA} \
      --trainer ${TRAINER} \
      --seed ${SEED} \
      --dataset_config "${PROJ_PATH}/configs/datasets/${DATASET}.yaml" \
      --trainer_config "${PROJ_PATH}/configs/trainers/${TRAINER}/${CFG}.yaml" \
      --output_dir ${OUTPUT_DIR} \
      --model_dir ${MODEL_DIR} \
      --load_epoch 5 \
      --eval_only \
      DATASET.NUM_SHOTS ${SHOTS} \
      MODEL.BACKBONE.PATH ${BACKBONE_PATH} \
      DATASET.SUBSAMPLE_CLASSES "new"
  fi
done