#!/bin/bash

##SBATCH -N 1
##SBATCH -n 5
##SBATCH -p gpu
##SBATCH --gres=gpu:1
##SBATCH --no-requeue
##SBATCH --output=run_parse.out
#
#source activate py37_coop

HOME_PATH="E:/work1_MuDPT/MuDPT"
#CFG="vit_b32_c16_ep200"
CTP="end"
CSC=False

#for DATASET in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101  ucf101 imagenet imagenet_a imagenet_r imagenet_sketch imagenetv2
for DATASET in ucf101
do
for CFG in vit_b16_c16_ep200 vit_b32_c16_ep200
do
  for SHOTS in 16 8 4 2 1
  do
      for KEY in accuracy macro_f1
      do
        LOG="${HOME_PATH}/scripts/coop/output/cls/${DATASET}/${CFG}_${SHOTS}shots/nctx_csc${CSC}_ctp${CTP}"
        OUTPUT="${HOME_PATH}/scripts/coop/parse/cls/${DATASET}/${CFG}_${SHOTS}shots_nctx16_csc${CSC}_ctp${CTP}/${KEY}"

        python ${HOME_PATH}/parse_test_res.py \
        --directory ${LOG}\
        --output_dir ${OUTPUT}\
        --keyword ${KEY} \
        --test_log
      done
  done
done
done

