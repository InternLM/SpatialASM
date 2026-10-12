bash Inference/edit_server.sh

#Start reasoning server
bash Inference/vllm_server.sh \
  "vllm--qwen-3.6-35BA3B" \
  "Qwen/Qwen3.6-35B-A3B" \
  "23333" \
  "1"

#Start captioning server
bash Inference/vllm_server.sh \
  "vllm--qwen-3.6-27B" \
  "Qwen/Qwen3.6-27B" \
  "23332" \
  "2"

IP=$(hostname -I | awk '{print $1}')

echo "Current IP: $IP"

# Wait for initializing VLLM servers
sleep 240
echo "Sleep complete!"

conda activate NVG # Not necessarily this environment (quite lenient for ASM_trajectory_construction.py)
python Data_Synthesis/ASM_trajectory_construction.py $IP
