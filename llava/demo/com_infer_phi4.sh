#!/bin/bash

# This script is used to infer the image with the model Phi-4.
QUESTION=$1
IMAGE_PATH=$2
MODEL_NAME_OR_PATH="microsoft/Phi-4"
VIT_PATH="openai/clip-vit-large-patch14-336/"
HLORA_PATH="/orange/chenaokun1990/tienyu/HealthGPT_pretrain/com_hlora_weights_phi4.bin"

python3 com_infer_phi4.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "32" \
    --hlora_alpha "64" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi4_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --question "$QUESTION" \
    --img_path "$IMAGE_PATH"
