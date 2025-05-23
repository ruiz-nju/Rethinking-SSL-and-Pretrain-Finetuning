#!/bin/bash

# custom config
DATA="path/to/data"
TRAINER=PromptSRC

DATASET=$1
SEED=$2
WEIGHTSPATH=$3

CFG=vit_b16_c2_ep20_batch4_4+4ctx
SHOTS=16
LOADEP=20
SUB_base=base
SUB_novel=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=${WEIGHTSPATH}/base/seed${SEED}
DIR_base=output/base2new/test_${SUB_base}/${COMMON_DIR}
DIR_novel=output/base2new/test_${SUB_novel}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Results are already available in ${DIR}. Skipping..."
else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"
    # Evaluate on base classes
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_base} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_base}

    # Evaluate on novel classes
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_novel} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_novel}

fi