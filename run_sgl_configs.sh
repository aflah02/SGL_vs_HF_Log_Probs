#!/bin/bash

ATTENTION_BACKENDS=("flashinfer" "fa3" "triton" "torch_native")

for backend in "${ATTENTION_BACKENDS[@]}"; do

    echo "Running backend=$backend without deterministic inference"
    CUDA_VISIBLE_DEVICES=1 python get_sgl_log_probs.py \
        --attention_backend "$backend"

    if [[ "$backend" == "torch_native" ]]; then
        echo "Skipping deterministic inference for backend=$backend (not supported)"
        continue
    fi

    echo "Running backend=$backend WITH deterministic inference"
    CUDA_VISIBLE_DEVICES=1 python get_sgl_log_probs.py \
        --attention_backend "$backend" \
        --enable_deterministic_inference
done