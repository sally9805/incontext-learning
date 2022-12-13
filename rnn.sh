#!/bin/bash
set -x

source consts.sh

DATA_NAME=$1
N_LAYERS=${2-4}
DATA_DIR=data/$DATA_NAME
mkdir -p $DATA_DIR

MODEL=LSTM
MODEL_SUFFIX=""
SEED=8
MODEL_DIR=rnn_output/${DATA_NAME}_SEED${SEED}_layer${N_LAYERS}
mkdir -p $MODEL_DIR


FULL_MODEL_NAME=${MODEL}${MODEL_SUFFIX}
python rnn/main.py \
        --cuda \
        --data $DATA_DIR \
        --epochs 10 \
        --save-interval 10 \
        --lr 1e-3 \
        --batch_size 8 \
        --bptt 1024 \
        --seed $SEED \
        --nlayers $N_LAYERS \
        --nhid 768 \
        --clip 1.0 \
        --model $MODEL \
        --save ${MODEL_DIR}/${FULL_MODEL_NAME}_checkpoint.pt

python rnn/generate.py \
    --data ${DATA_DIR} \
    --checkpoint ${MODEL_DIR}/${FULL_MODEL_NAME}_checkpoint.pt \
    --cuda \
    --results_name ${FULL_MODEL_NAME}_in_context_results.tsv
