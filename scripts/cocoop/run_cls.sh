#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=run_cls.out

source activate py37_coop

PROJ_PATH="/data01/home/scy0037/run/myz"
DATA="${PROJ_PATH}/datasets"
TRAINER="CoCoOp"

CFG="vit_b32_bz1_ep10_ctxv1"
BACKBONE_PATH="${PROJ_PATH}/MuDPT/pretrained/ViT-B-32.pt"
CTP="end"
CSC=False
NCTX=4
SHOTS=16
SEED=2

for DATASET in imagenet
do
  DIR="${PROJ_PATH}/output/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}"

  if [ -d "$DIR" ]; then
     echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
     python ../../train.py \
            --dataset_root ${DATA} \
            --output_dir ${DIR} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --trainer_config "${PROJ_PATH}/MuDPT/configs/trainers/${TRAINER}/${CFG}.yaml" \
            --dataset_config "${PROJ_PATH}/MuDPT/configs/datasets/${DATASET}.yaml" \
            TRAINER.COCOOP.N_CTX ${NCTX} \
            TRAINER.COCOOP.CSC ${CSC} \
            TRAINER.COCOOP.CLASS_TOKEN_POSITION ${CTP} \
            MODEL.BACKBONE.PATH ${BACKBONE_PATH} \
            DATASET.NUM_SHOTS ${SHOTS}
  fi
done