#!/bin/bash
# Usage: bash scripts/knockout/worldsense/run_baseline.sh
TASK=worldsense_shorts
MODEL=qwen2_5_omni_knockout

cd "$(dirname "$0")/../../.."               # repo root
export PYTHONPATH=".:lmms-eval:$PYTHONPATH"
export FPS_MAX_FRAMES=128
export KNOCKOUT_MODEL_SIZE=3B
export LMM_EVAL_SAVE_TYPE=baseline
export LMM_EVAL_OUTPUT_DIR=logs/knockout/Qwen2.5-Omni-3B/$TASK/
mkdir -p "$LMM_EVAL_OUTPUT_DIR"

accelerate launch --num_processes=1 -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=Qwen/Qwen2.5-Omni-3B,attn_implementation=eager" \
    --tasks $TASK --batch_size 1 --log_samples \
    --log_samples_suffix "knockout_baseline_3B" \
    --output_path "$LMM_EVAL_OUTPUT_DIR"
