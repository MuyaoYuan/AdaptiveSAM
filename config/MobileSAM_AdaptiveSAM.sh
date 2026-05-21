#!/usr/bin/env bash

set -x
PY_ARGS=${@:1}

python -u main.py \
    --data-path=./data/CamVid \
    --split=val \
    --save-path=./exp/CamVid/val/MobileSAM_AdaptiveSAM \
    --batch-size=1 \
    --ref_gap=12 \
    --backend efficient_vit_t \
    --adapter_type=4 \
    --adapter_option -1 0 1 2 \
    --no_multimask \
    --search_type evolution \
    --population-num=50 \
    --subnet_weight=retrain \
    --num_entry=10 \
    --bs_lookup_init=10 \
    --indicator_name SC \
    --sc_lamda=1 \
    --num_M=256 \
    --early_feat=True \
    --stream=True \
    --neg_loader camvid \
    --num_pos_samples=1 \
    --num_neg_samples=19 \
    --cache_capacity=1 \
    ${PY_ARGS}
