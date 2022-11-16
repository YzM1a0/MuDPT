#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=vpt_imagenet.out

source activate py37_coop

PROJ_PATH="/data01/home/scy0037/run/myz/MuDPT"
DATA="${PROJ_PATH}/datasets"
TRAINER="VPT"
BACKBONE_PATH="${PROJ_PATH}/pretrained/ViT-B-16.pt"
CFG="vit_b16_c2_ep5_batch4"
IMG_N_CTX=8
DATASET="imagenet"
SHOTS=16
SEED=2

DIR=output/cls/${DATASET}/${CFG}_${SHOTS}shots/nctx${IMG_N_CTX}_depth12/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python ${PROJ_PATH}/train.py \
    --dataset_root ${DATA} \
    --output_dir ${DIR} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --trainer_config "${PROJ_PATH}/configs/trainers/${TRAINER}/${CFG}.yaml" \
    --dataset_config "${PROJ_PATH}/configs/datasets/${DATASET}.yaml" \
    MODEL.BACKBONE.PATH ${BACKBONE_PATH} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES "all"
fi