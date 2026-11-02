#!/bin/bash
# Usage: bash scripts/drop_token/av_speakerbench/run_baseline.sh
# Baseline with no tokens dropped (the Baseline row of Table 4).
TASK=av_speakerbench_audiovisual
MODEL=qwen2_5_omni
OUTPUT_DIR=logs/eval_outputs/$MODEL/$TASK/baseline

cd "$(dirname "$0")/../../.."               # repo root
export PYTHONPATH=".:lmms-eval:$PYTHONPATH"
export FPS_MAX_FRAMES=128
export LMM_EVAL_OUTPUT_DIR=""               # required, do not remove

mkdir -p "$OUTPUT_DIR"
accelerate launch --num_processes=1 -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=Qwen/Qwen2.5-Omni-3B,attn_implementation=eager,exp_config=config/$MODEL/$TASK/eager/baseline.yaml" \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "eval_${MODEL}_${TASK}" \
    --output_path "$OUTPUT_DIR"
