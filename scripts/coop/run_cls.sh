#!/bin/bash

DATA="D:/Datasets/VisualClassification"
TRAINER="CoOp"

CFG="vit_b16_ep50_ctxv1"
BACKBONE_PATH="../../pretrained/ViT-B-16.pt"
CTP="end"
NCTX=16
CSC=False

#for DATASET in imagenet sun397
# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101  ucf101
for DATASET in caltech101
do
for SHOTS in 1 2 4 8 16
  do
    for SEED in 1 2 3
    do
        DIR=output/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python ../../train.py \
            --dataset_root ${DATA} \
            --output_dir ${DIR} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --trainer_config ../../configs/trainers/${TRAINER}/${CFG}.yaml \
            --dataset_config ../../configs/datasets/${DATASET}.yaml \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            MODEL.BACKBONE.PATH ${BACKBONE_PATH} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
    done
  done
done