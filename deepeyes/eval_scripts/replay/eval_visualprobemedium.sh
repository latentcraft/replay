export HF_HOME=/root/cache/huggingface
export MODEL_PATH=ChenShawn/DeepEyes-7B
export TIMESTAMP=T260412_173328
export CHUNK_IDX="${1:-0}"
# Always --dev: at most 4 samples; use chunk_idx 0 only (see replay_eval).

python replay_eval/eval_visualprobemedium.py \
    --model-path "$MODEL_PATH" \
    --output-prefix ../OUTPUT/DeepEyes/outputs/replay/DeepEyes-7B/visualprobemedium/$TIMESTAMP \
    --chunk_idx "$CHUNK_IDX" \
    --dev
