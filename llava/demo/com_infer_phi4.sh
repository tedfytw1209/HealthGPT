#!/bin/bash

# This script is used to infer the image with the model Phi-4.
QUESTION=$1
IMAGE_PATH=$2
MODEL_NAME_OR_PATH="/orange/guoj1/tienyuchang/huggingface/hub/models--microsoft--phi-4/snapshots/187ef0342fff0eb3333be9f00389385e95ef0b61/"
VIT_PATH="/orange/guoj1/tienyuchang/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/"
HLORA_PATH="/orange/guoj1/tienyuchang/HealthGPT_L14/com_hlora_weights_phi4.bin"

python3 llava/demo/com_infer_phi4.py \
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
