#!/bin/bash
# Usage: bash scripts/knockout/salmonn_av_speakerbench/run_knockout.sh <config_id> [range]
#   e.g. bash scripts/knockout/salmonn_av_speakerbench/run_knockout.sh 1 0-11
# Run run_baseline.sh first. Config ids: 1-21, 31, 32. Centers: 0-35.
CONFIG=${1:?give a config id}
RANGE=${2:-0-35}
WINDOW_K=7
TASK=av_speakerbench_audiovisual
MODEL=video_salmonn_2_full_knockout

cd "$(dirname "$0")/../../.."               # repo root
export PYTHONPATH=".:models/video-SALMONN-2/video_SALMONN2_plus:lmms-eval:$PYTHONPATH"
export VIDEO_SALMONN_REPO=models/video-SALMONN-2/video_SALMONN2_plus
export FPS_MAX_FRAMES=128
export KNOCKOUT_MODEL_SIZE=3B
export LMM_EVAL_SAVE_TYPE=knockout
export LMM_EVAL_OUTPUT_DIR=logs/knockout/Video-SALMONN2_plus_3B_full/$TASK/
export KNOCKOUT_CONFIG_IDS=$CONFIG KNOCKOUT_SWEEP_TYPE=window
export KNOCKOUT_WINDOW_K=$WINDOW_K KNOCKOUT_SAMPLE_MODE=correct
mkdir -p "$LMM_EVAL_OUTPUT_DIR"

for i in $(seq "${RANGE%-*}" "${RANGE#*-}"); do
    KNOCKOUT_SWEEP_RANGE=$i-$i accelerate launch --num_processes=1 -m lmms_eval \
        --model $MODEL \
        --model_args "pretrained=tsinghua-ee/video-SALMONN2_plus_3B_full,attn_implementation=eager,max_num_frames=128,max_pixels=25088,base_interval=0.5" \
        --tasks $TASK --batch_size 1 --log_samples \
        --log_samples_suffix "knockout_knockout_3B" \
        --output_path "$LMM_EVAL_OUTPUT_DIR"
done
