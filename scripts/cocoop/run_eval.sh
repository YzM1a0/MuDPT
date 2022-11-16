#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=run_coop_eval.out

source activate py37_coop

PROJ_PATH="E:/work1_MuDPT/MuDPT"
DATA="E:/Datasets/VisualClassification"
TRAINER="CoCoOp"

CFG="vit_b32_bz1_ep10_ctxv1"
BACKBONE_PATH="${PROJ_PATH}/pretrained/ViT-B-32.pt"
CTP="end"
SHOTS=16
NCTX=4
CSC=False

for DATASET in imagenet imagenet_sketch imagenet_r imagenet_a imagenetv2
do
  for SEED in 1 2 3
  do
    python ../../train.py \
    --dataset_root ${DATA} \
    --output_dir output/evaluation/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --seed ${SEED} \
    --trainer_config "${PROJ_PATH}/configs/trainers/${TRAINER}/${CFG}.yaml" \
    --dataset_config "${PROJ_PATH}/configs/datasets/${DATASET}.yaml" \
    --trainer ${TRAINER} \
    --eval_only \
    --model_dir "output/imagenet/${CFG}_16shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed2" \
    --load_epoch=50 \
    MODEL.BACKBONE.PATH ${BACKBONE_PATH}
  done
done