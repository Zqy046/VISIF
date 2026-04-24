#!/bin/bash

model_name=AutoTimes_InternVL
station=${1:-CNR}  # Default to CNR if not provided

echo station

# Data paths (adjust these paths according to your setup)
DATA_DIR=""
STATS_PATH="stats_multi.json"
TIMESTAMP_DIR=""
CHECKPOINT_DIR="./checkpoints"
INTERNVL_MODEL_DIR=""

YEARS_TRAIN="2008_nonhrv 2009_nonhrv 2010_nonhrv 2011_nonhrv 2012_nonhrv 2013_nonhrv 2014_nonhrv 2015_nonhrv 2016_nonhrv"
YEARS_VAL="2017_nonhrv 2018_nonhrv 2019_nonhrv"
YEARS_TEST="2020_nonhrv 2021_nonhrv 2022_nonhrv"

torchrun --nnodes 1 --nproc-per-node 4 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id "cvv_tscontext_${station}_48_8" \
  --model $model_name \
  --data cvv_tscontext \
  --data_dir $DATA_DIR \
  --stats_path $STATS_PATH \
  --timestamp_dir $TIMESTAMP_DIR \
  --timestamp_range "20082022" \
  --context_channels "IR_039" "IR_087" "IR_108" "VIS006" "VIS008" "WV_062" "WV_073" \
  --optflow_channels "IR_039_vx" "IR_039_vy" "IR_087_vx" "IR_087_vy" "IR_108_vx" "IR_108_vy" "WV_062_vx" "WV_062_vy" "WV_073_vx" "WV_073_vy" \
  --ts_channels "GHI" \
  --ts_target_channels "GHI" \
  --years_train $YEARS_TRAIN \
  --years_val $YEARS_VAL \
  --years_test1 $YEARS_TEST \
  --years_test2 $YEARS_TEST \
  --years_test3 $YEARS_TEST \
  --stations_train "PCCI_20082022_${station}" \
  --stations_val "PCCI_20082022_${station}" \
  --stations_test1 "PCCI_20082022_IZA" \
  --stations_test2 "PCCI_20082022_CNR" \
  --stations_test3 "PCCI_20082022_PAL" \
  --image_size 112 112 \
  --seq_len 48 \
  --label_len 40 \
  --token_len 8 \
  --test_seq_len 48 \
  --test_label_len 40 \
  --test_pred_len 8 \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --patience 15 \
  --use_amp \
  --mlp_hidden_dim 512 \
  --mlp_hidden_layers 2 \
  --mlp_activation relu \
  --use_multi_gpu \
  --mix_embeds \
  --llm_ckp_dir $INTERNVL_MODEL_DIR \
  --checkpoints $CHECKPOINT_DIR \
  --freq "t" \
  --num_workers 16 \
  --des "CVV_TSContext_${station}_Exp_TrainValLastToken_InternVL2-2B"
