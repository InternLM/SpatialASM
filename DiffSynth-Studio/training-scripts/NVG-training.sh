#!/bin/bash
NAME="SpatialASM-NVG-traing"
ROOT_DIR="xxx" #Change to the root directory of all training images.
Output_Path="/Gen/SpatialNVG-LoRA"
export DIFFSYNTH_SKIP_DOWNLOAD=True

accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  examples/flux2/model_training/train.py \
  --dataset_base_path "${ROOT_DIR}" \
  --dataset_metadata_path "training-scripts/training_samples.jsonl" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_paths '[
    [
        "black-forest-labs/FLUX.2-klein-9B/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "black-forest-labs/FLUX.2-klein-9B/transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
    ],
    [
        "black-forest-labs/FLUX.2-klein-9B/text_encoder/model-00001-of-00004.safetensors",
        "black-forest-labs/FLUX.2-klein-9B/text_encoder/model-00002-of-00004.safetensors",
        "black-forest-labs/FLUX.2-klein-9B/text_encoder/model-00003-of-00004.safetensors",
        "black-forest-labs/FLUX.2-klein-9B/text_encoder/model-00004-of-00004.safetensors"
    ],
    "black-forest-labs/FLUX.2-klein-9B/vae/diffusion_pytorch_model.safetensors"
  ]' \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-9B/tokenizer"\
  --learning_rate 1e-4 \
  --num_epochs 3 \
  --save_steps 12000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${Output_Path}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out" \
  --lora_rank 512 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 >"${NAME}.txt" 2>&1
