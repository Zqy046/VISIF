#!/bin/bash

stations=(
  "IZA"
  "CNR"
  "PAL"
)

models=(
  "InternVL2-4B"
  "InternVL2-8B"
)

# preprocess timestamps to generate text embedding with InternVL
for station in "${stations[@]}"; do
  for model in "${models[@]}"; do
    python preprocess_cvv/preprocess4cvv_tscontext_InternVL.py --gpu 0 --dataset_time_range "20082022" --site "$station" \
     --llm_ckp_dir "./InternVL/${model}/" \
     --save_dir_path "./data_stamp/${model}/20082022"
  done
done
