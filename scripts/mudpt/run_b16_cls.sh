#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu 
#SBATCH --gres=gpu:2
#SBATCH --no-requeue
#SBATCH --output=run_b16_cls.out

source activate py37_coop

PROJ_PATH="/data/home/scy0037/run/myz/MuDPT"
DATA="${PROJ_PATH}/datasets"
TRAINER="MuDPT"

CFG="vit_b16_bz4_ep10_nctx4_depth9"
BACKBONE_PATH="${PROJ_PATH}/pretrained/ViT-B-16.pt"
N_CTX=4
DEEP_PROMPT_DEPTH=9

for DATASET in caltech101 oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 ucf101 imagenet sun397
do
for SHOTS in 16 8 4 2 1
do
  for SEED in 3 2 1
  do
    DIR=output/cls/${DATASET}/${CFG}_${SHOTS}shots/nctx${N_CTX}_depth${DEEP_PROMPT_DEPTH}/seed${SEED}

    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python ../../train.py \
        --dataset_root ${DATA} \
        --output_dir ${DIR} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --trainer_config "${PROJ_PATH}/configs/trainers/${TRAINER}/${CFG}.yaml" \
        --dataset_config "${PROJ_PATH}/configs/datasets/${DATASET}.yaml" \
        MODEL.BACKBONE.PATH ${BACKBONE_PATH}
      fi
  done
done
done