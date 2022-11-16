#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=base2new_new.out

source activate py37_coop

PROJ_PATH="/data01/home/scy0037/run/myz"
DATA="${PROJ_PATH}/datasets"
TRAINER="CoOp"
CFG="vit_b32_ep50"
BACKBONE_PATH="${PROJ_PATH}/MuDPT/pretrained/ViT-B-32.pt"
N_CTX=4
SHOTS=16
SEED=2

#
for DATASET in oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 caltech101 ucf101 imagenet sun397 oxford_pets
do
  MODEL_DIR=output/base2new/base/${DATASET}/shots_${SHOTS}/${CFG}_nctx${N_CTX}/seed${SEED}
  DIR=output/base2new/new/${DATASET}/shots_${SHOTS}/${CFG}_nctx${N_CTX}/seed${SEED}
  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
      python ../../train.py \
      --dataset_root ${DATA} \
      --trainer ${TRAINER} \
      --seed ${SEED} \
      --dataset_config "${PROJ_PATH}/MuDPT/configs/datasets/${DATASET}.yaml" \
      --trainer_config "${PROJ_PATH}/MuDPT/configs/trainers/${TRAINER}/${CFG}.yaml" \
      --output_dir ${DIR} \
      --model_dir ${MODEL_DIR} \
      --eval_only \
      TRAINER.COOP.N_CTX ${N_CTX} \
      DATASET.NUM_SHOTS ${SHOTS} \
      MODEL.BACKBONE.PATH ${BACKBONE_PATH} \
      DATASET.SUBSAMPLE_CLASSES "new"
  fi
done