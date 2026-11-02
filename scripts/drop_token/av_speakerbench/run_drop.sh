#!/bin/bash
# Usage: bash scripts/drop_token/av_speakerbench/run_drop.sh [method]
# Methods: tokendrop_video26_audio26_question29 (default),
#          tokendrop_video26, tokendrop_audio26, tokendrop_question29
METHOD=${1:-tokendrop_video26_audio26_question29}
TASK=av_speakerbench_audiovisual
MODEL=qwen2_5_omni
OUTPUT_DIR=logs/eval_outputs/$MODEL/$TASK/$METHOD

cd "$(dirname "$0")/../../.."               # repo root
export PYTHONPATH=".:lmms-eval:$PYTHONPATH"
export FPS_MAX_FRAMES=128
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
