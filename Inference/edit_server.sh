#!/usr/bin/env bash
set -e


mkdir -p Inference/logs
conda activate NVG

BASE_PORT=8000
GPU_ID=0

CUDA_VISIBLE_DEVICES=${GPU_ID} \
nohup python3 Inference/edit_server.py \
    --base ${BASE_PORT} \
    > Inference/logs/edit_worker_${GPU_ID}.log 2>&1 &
echo ${BASE_PORT}
echo $! > Inference/logs/edit_worker_${GPU_ID}.pid

echo "worker started, pid=$!"
