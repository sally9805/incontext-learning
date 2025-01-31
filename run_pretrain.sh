#!/bin/bash
set -x

source consts.sh

DATA_NAME=$1
DATA_DIR=data/$DATA_NAME
OUTPUT_DIR=outputs_medium

SEED=$2
EPOCHS=${3-5}
LR=${4-8e-4}
WARMUP=${5-1000}


python run_clm.py \
    --model_type gpt2 \
    --tokenizer_name $DATA_DIR/tokenizer.json \
    --small_model \
    --custom_tokenizer \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --output_dir $OUTPUT_DIR/${DATA_NAME}_SEED${SEED}_pretrain \
    --logging_steps 100 \
    --save_total_limit 4 \
    --evaluation_strategy steps \
    --save_steps 1500 \
    --seed $SEED \
    --fp16 \
    --warmup_steps ${WARMUP} \
    --lr_scheduler_type linear \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --custom_num_layers 12
