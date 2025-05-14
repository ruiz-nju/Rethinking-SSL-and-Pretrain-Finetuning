#!/bin/bash

#cd ../..

# custom config
DATA="path/to/data"
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
SUB=$4
CSC=False  # class-specific context (False or True)
DEVICE=$5

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_${SUB}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.NUM_LABELED ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done