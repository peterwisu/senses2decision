#!/bin/bash
# Usage: bash scripts/drop_token/av_odyssey/run_drop.sh [method]
# Methods: tokendrop_targets25_query31_question_29 (default),
#          tokendrop_targets25, tokendrop_query31, tokendrop_question29
METHOD=${1:-tokendrop_targets25_query31_question_29}
TASK=av_odyssey
MODEL=qwen2_5_omni_interleave
OUTPUT_DIR=logs/eval_outputs/$MODEL/$TASK/$METHOD

cd "$(dirname "$0")/../../.."               # repo root
export PYTHONPATH=".:lmms-eval:$PYTHONPATH"
export FPS_MAX_FRAMES=768
export LMM_EVAL_OUTPUT_DIR=""               # required, do not remove

mkdir -p "$OUTPUT_DIR"
accelerate launch --num_processes=1 -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=Qwen/Qwen2.5-Omni-3B,attn_implementation=eager,exp_config=config/$MODEL/$TASK/eager/$METHOD.yaml" \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "eval_${MODEL}_${TASK}" \
    --output_path "$OUTPUT_DIR"
