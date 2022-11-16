#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=xd_test.out

source activate py37_coop

PROJ_PATH="/data01/home/scy0037/run/myz/MuDPT"
DATA="${PROJ_PATH}/datasets"
TRAINER="VPT"
BACKBONE_PATH="${PROJ_PATH}/pretrained/ViT-B-16.pt"
CFG="vit_b16_c2_ep5_batch4"
IMG_N_CTX=8
SHOTS=16
SEED=2


for DATASET in caltech101 oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 ucf101
do
  MODEL_DIR="output/cls/imagenet/${CFG}_${SHOTS}shots/nctx${IMG_N_CTX}_depth12/seed${SEED}"
  OUTPUT_DIR="output/xd_evaluation/${DATASET}/${CFG}_${SHOTS}shots/nctx${IMG_N_CTX}_depth12/seed${SEED}"

  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
    python ${PROJ_PATH}/MuDPT/train.py \
    --dataset_root ${DATA} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --trainer_config "${PROJ_PATH}/configs/trainers/${TRAINER}/${CFG}.yaml" \
    --dataset_config "${PROJ_PATH}/configs/datasets/${DATASET}.yaml" \
    --trainer ${TRAINER} \
    --eval_only \
    --model_dir ${MODEL_DIR} \
    --load_epoch 5 \
    MODEL.BACKBONE.PATH ${BACKBONE_PATH}
  fi
done