#!/bin/bash
# Usage: bash scripts/knockout/salmonn_worldsense/run_baseline.sh
TASK=worldsense_shorts
MODEL=video_salmonn_2_full_knockout

cd "$(dirname "$0")/../../.."               # repo root
export PYTHONPATH=".:models/video-SALMONN-2/video_SALMONN2_plus:lmms-eval:$PYTHONPATH"
export VIDEO_SALMONN_REPO=models/video-SALMONN-2/video_SALMONN2_plus
export FPS_MAX_FRAMES=128
export KNOCKOUT_MODEL_SIZE=3B
export LMM_EVAL_SAVE_TYPE=baseline
export LMM_EVAL_OUTPUT_DIR=logs/knockout/Video-SALMONN2_plus_3B_full/$TASK/
mkdir -p "$LMM_EVAL_OUTPUT_DIR"

accelerate launch --num_processes=1 -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=tsinghua-ee/video-SALMONN2_plus_3B_full,attn_implementation=eager,max_num_frames=128,max_pixels=25088,base_interval=0.5" \
    --tasks $TASK --batch_size 1 --log_samples \
    --log_samples_suffix "knockout_baseline_3B" \
    --output_path "$LMM_EVAL_OUTPUT_DIR"
