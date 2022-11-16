#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --no-requeue
#SBATCH --output=run_coop_eval.out

source activate py37_coop

PROJ_PATH="/data01/home/scy0037/run/myz/MuDPT"
DATA="${PROJ_PATH}/datasets"
TRAINER="VPT"

CFG="vit_b16_c2_ep5_batch4"
BACKBONE_PATH="${PROJ_PATH}/pretrained/ViT-B-16.pt"
DEEP_IMG_N_CTX=8
IMG_PROMPT_DEPTH=12
SHOTS=16
SEED=2

for DATASET in imagenet imagenet_sketch imagenet_r imagenet_a imagenetv2
do
  for SEED in 1
  do
    python ../../train.py \
    --dataset_root ${DATA} \
    --output_dir output/evaluation/${DATASET}/${CFG}_${SHOTS}shots/nctx${DEEP_IMG_N_CTX}_depth${IMG_PROMPT_DEPTH}/seed${SEED} \
    --seed ${SEED} \
    --trainer_config "${PROJ_PATH}/configs/trainers/${TRAINER}/${CFG}.yaml" \
    --dataset_config "${PROJ_PATH}/configs/datasets/${DATASET}.yaml" \
    --trainer ${TRAINER} \
    --eval_only \
    --model_dir "output/cls/imagenet/${CFG}_${SHOTS}shots/nctx${DEEP_IMG_N_CTX}_depth${IMG_PROMPT_DEPTH}/seed2" \
    --load_epoch 5 \
    MODEL.BACKBONE.PATH ${BACKBONE_PATH}
  done
done