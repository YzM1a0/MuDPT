#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=run_coop_eval.out

source activate py37_coop

PROJ_PATH="/data01/home/scy0037/run/myz"
DATA="${PROJ_PATH}/datasets"
TRAINER="CoOp"

CFG="vit_b16_ep50"
BACKBONE_PATH="${PROJ_PATH}/MuDPT/pretrained/ViT-B-16.pt"
CTP="end"
SHOTS=16
NCTX=4
CSC=False
SEED=2

for DATASET in caltech101 oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 ucf101
do
  MODEL_DIR="${PROJ_PATH}/MuDPT/scripts/coop/output/imagenet/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}"
  OUTPUT_DIR="{PROJ_PATH}/MuDPT/scripts/coop/output/xd_evaluation/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}"

  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
    python ${PROJ_PATH}/MuDPT/train.py \
    --dataset_root ${DATA} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --trainer_config "${PROJ_PATH}/MuDPT/configs/trainers/${TRAINER}/${CFG}.yaml" \
    --dataset_config "${PROJ_PATH}/MuDPT/configs/datasets/${DATASET}.yaml" \
    --trainer ${TRAINER} \
    --eval_only \
    --model_dir ${MODEL_DIR} \
    --load_epoch=50 \
    MODEL.BACKBONE.PATH ${BACKBONE_PATH}
  fi
done