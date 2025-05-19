#!/bin/bash

model_name=$1
log_dir=$2

python3 tmlu_eval.py \
    --backend vllm \
    --model $model_name \
    --dtype bfloat16 \
    --temperature 0.0 \
    --max_length 8192 \
    --max_tokens 4096 \
    --subsets ALL \
    --tensor_parallel_size 1 \
    --log_dir $log_dir \
    --few_shot_num 5 \
    --cot