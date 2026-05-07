export HF_HOME=/mnt/bn/genai-global-apply-ml/yun.xing/cache/huggingface
export DECORD_EOF_RETRY_MAX=20480
export MODEL_PATH="${MODEL_PATH:-Video-R1/Video-R1-7B}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
export PROCESSOR_MODEL_PATH="${PROCESSOR_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
export TIMESTAMP=T260412_173328
export CHUNK_IDX="${1:-0}"
# Default: full eval. Smoke test: bash eval_videommmu.sh 0 --dev
if [ "$#" -ge 1 ]; then shift; fi

python replay_eval/eval_videommmu.py \
    --model-path "$MODEL_PATH" \
    --base-model-path "$BASE_MODEL_PATH" \
    --processor-model-path "$PROCESSOR_MODEL_PATH" \
    --output-prefix ../OUTPUT/Video-R1/replay/Video-R1-7B/videommmu/"$TIMESTAMP" \
    --chunk_idx "$CHUNK_IDX" \
    --dev
