#!/bin/bash
# Usage: bash scripts/knockout/av_odyssey/run_knockout.sh <config_id> [range]
#   e.g. bash scripts/knockout/av_odyssey/run_knockout.sh 1 0-11
# Run run_baseline.sh first. Config ids: 1-15. Centers: 0-35.
CONFIG=${1:?give a config id}
RANGE=${2:-0-35}
WINDOW_K=7
TASK=av_odyssey
MODEL=qwen2_5_omni_interleave_knockout

cd "$(dirname "$0")/../../.."               # repo root
export PYTHONPATH=".:lmms-eval:$PYTHONPATH"
export FPS_MAX_FRAMES=128
export KNOCKOUT_MODEL_SIZE=3B
export LMM_EVAL_SAVE_TYPE=knockout
export LMM_EVAL_OUTPUT_DIR=logs/knockout/Qwen2.5-Omni-3B/$TASK/
export KNOCKOUT_CONFIG_IDS=$CONFIG KNOCKOUT_SWEEP_TYPE=window
export KNOCKOUT_WINDOW_K=$WINDOW_K KNOCKOUT_SAMPLE_MODE=correct
mkdir -p "$LMM_EVAL_OUTPUT_DIR"

for i in $(seq "${RANGE%-*}" "${RANGE#*-}"); do
    KNOCKOUT_SWEEP_RANGE=$i-$i accelerate launch --num_processes=1 -m lmms_eval \
        --model $MODEL \
        --model_args "pretrained=Qwen/Qwen2.5-Omni-3B,attn_implementation=eager" \
        --tasks $TASK --batch_size 1 --log_samples \
        --log_samples_suffix "knockout_knockout_3B" \
        --output_path "$LMM_EVAL_OUTPUT_DIR"
done
