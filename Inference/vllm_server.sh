#!/usr/bin/env bash
set -e

mkdir -p Inference/logs
conda activate spatialasm_vllm


SERVED_MODEL_NAME="${1:?请传入 served model name}"
MODEL_PATH="${2:?请传入 model 路径}"
PORT="${3:?请传入端口号}"
GPU_ID="${4:?请传入 GPU ID}"

echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "GPU ID: ${GPU_ID}"
echo "Start to run vLLM server"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
nohup python -m vllm.entrypoints.openai.api_server \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --model "${MODEL_PATH}" \
  --port "${PORT}" \
  --gpu-memory-utilization 0.93 \
  --trust-remote-code \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  > "Inference/logs/${SERVED_MODEL_NAME}.log" 2>&1 &

PID=$!
echo "${PID}" > "Inference/logs/${SERVED_MODEL_NAME}.pid"

echo "vLLM server started, PID: ${PID}"
echo "Log: Inference/logs/${SERVED_MODEL_NAME}.log"