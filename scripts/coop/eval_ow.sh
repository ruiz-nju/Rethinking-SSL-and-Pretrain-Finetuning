#!/bin/bash

#cd ../..

# custom config
DATA="path/to/data"
TRAINER=CoOp
NCTX=16
CSC=False
CTP=end

DATASET=$1
CFG=$2
SHOTS=$3
SUB=$4
DEVICE=$5


for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_${SUB}/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_base/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch 50 \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
done