#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=run_parse.out

source activate py37_coop

PROJ_PATH="/data/home/scy0037/run/myz/MuDPT"
CFG="vit_b16_bz4_ep10_nctx4_depth9"
N_CTX=4
DEEP_PROMPT_DEPTH=9


for DATASET in caltech101 oxford_pets oxford_flowers fgvc_aircraft
do
  for SHOTS in 16 8 4 2 1
  do
      for KEY in accuracy macro_f1
      do
        LOG="${PROJ_PATH}/scripts/mudpt/output/cls/${DATASET}/${CFG}_${SHOTS}shots/nctx${N_CTX}_depth${DEEP_PROMPT_DEPTH}"
        OUTPUT="${PROJ_PATH}/scripts/mudpt/parse/cls/${DATASET}/${CFG}_${SHOTS}shots/nctx${N_CTX}_depth${DEEP_PROMPT_DEPTH}"

        python ${HOME_PATH}/parse_test_res.py \
        --directory ${LOG}\
        --output_dir ${OUTPUT}\
        --keyword ${KEY} \
        --test_log
      done
  done

done

