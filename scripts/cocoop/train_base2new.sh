#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --no-requeue
#SBATCH --output=train_base2new.out

source activate py37_coop

PROJ_PATH="/data01/home/scy0037/run/myz/MuDPT"
DATA="${PROJ_PATH}/datasets"
TRAINER="CoCoOp"
CFG="vit_b32_bz1_ep10_ctxv1"
BACKBONE_PATH="${PROJ_PATH}/pretrained/ViT-B-32.pt"
N_CTX=4
SHOTS=16
SEED=2

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 
for DATASET in caltech101 ucf101 imagenet sun397
do
  DIR=output/base2new/base/${DATASET}/shots_${SHOTS}/${CFG}_nctx${N_CTX}/seed${SEED}

  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
      python ../../train.py \
      --dataset_root ${DATA} \
      --trainer ${TRAINER} \
      --seed ${SEED} \
      --dataset_config "${PROJ_PATH}/configs/datasets/${DATASET}.yaml" \
      --trainer_config "${PROJ_PATH}/configs/trainers/${TRAINER}/${CFG}.yaml" \
      --output_dir ${DIR} \
      TRAINER.COCOOP.N_CTX ${N_CTX} \
      DATASET.NUM_SHOTS ${SHOTS} \
      MODEL.BACKBONE.PATH ${BACKBONE_PATH} \
      DATASET.SUBSAMPLE_CLASSES "base"
  fi
done