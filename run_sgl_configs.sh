#!/bin/bash

# Backends taken from https://docs.sglang.io/advanced_features/attention_backend.html#launch-command-for-different-attention-backends. Only these 3 backends support both deterministic inference
ATTENTION_BACKENDS=("flashinfer" "fa3" "triton")

for backend in "${ATTENTION_BACKENDS[@]}"; do

    echo "Running backend=$backend without deterministic inference"
    CUDA_VISIBLE_DEVICES=1 python get_sgl_log_probs.py \
        --attention_backend "$backend"

    echo "Running backend=$backend WITH deterministic inference"
    CUDA_VISIBLE_DEVICES=1 python get_sgl_log_probs.py \
        --attention_backend "$backend" \
        --enable_deterministic_inference

done