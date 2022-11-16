#!/bin/bash

DATA="D:/Datasets/VisualClassification"
TRAINER="CoOp"

CFG="vit_b16_ep50_ctxv1"
BACKBONE_PATH="../../pretrained/ViT-B-16.pt"
CTP="end"
SHOTS=16
NCTX=4
CSC=False

for DATASET in imagenet_sketch imagenet_r imagenet_a imagenetv2
do
  for SEED in 1 2 3
  do
    python ../../train.py \
    --dataset_root ${DATA} \
    --output_dir output/DG/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --seed ${SEED} \
    --trainer_config "../configs/trainers/${TRAINER}/${CFG}.yaml" \
    --dataset_config "../configs/datasets/${DATASET}.yaml" \
    --trainer ${TRAINER} \
    --eval_only \
    --model_dir "output/imagenet/vit_b16_ep50_16shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed2" \
    --load_epoch=50 \
    MODEL.BACKBONE.PATH ${BACKBONE_PATH}
  done
done