from fastapi import FastAPI, Response
from pydantic import BaseModel
from PIL import Image
import uvicorn, torch, os, traceback
import threading

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import argparse

# Change all the Path 'YuhongLiu/SpatialASM-NVG' to your local path.
transformer_path = ["YuhongLiu/SpatialASM-NVG/transformer/diffusion_pytorch_model.safetensors"]
text_encoder_path = ["YuhongLiu/SpatialASM-NVG/text_encoder/model-00001-of-00004.safetensors", "YuhongLiu/SpatialASM-NVG/text_encoder/model-00002-of-00004.safetensors", "YuhongLiu/SpatialASM-NVG/text_encoder/model-00003-of-00004.safetensors", "YuhongLiu/SpatialASM-NVG/text_encoder/model-00004-of-00004.safetensors"]
vae_path = "YuhongLiu/SpatialASM-NVG/vae/diffusion_pytorch_model.safetensors"


app = FastAPI()
pipe = None
infer_lock = threading.Lock()


@app.on_event("startup")
def load_model():
    global pipe
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=transformer_path),
            ModelConfig(path=text_encoder_path),
            ModelConfig(path=vae_path),
        ],
        tokenizer_config=ModelConfig(path="YuhongLiu/SpatialASM-NVG/tokenizer"),
    )
    print("Flux2 pipeline loaded.")

class EditRequest(BaseModel):
    input_paths: list[str]
    prompt: str
    seed: int = 42
    num_inference_steps: int = 25
    cfg_scale: float = 10.0
    save_folder: str = "."
    num: int = 0
    cnt: int = 0

@app.post("/edit", status_code=204)
def edit(req: EditRequest):
    try:
        save_folder = req.save_folder
        num = req.num
        cnt = req.cnt
        images = [Image.open(p).convert("RGB") for p in req.input_paths]
        with infer_lock:  
            out = pipe(
                req.prompt,
                edit_image=images,
                seed=req.seed,
                rand_device="cuda",
                num_inference_steps=req.num_inference_steps,
                cfg_scale=req.cfg_scale,
            )
        out.save(f"{save_folder}/edited_{num}_{cnt}.png")
        print(f"saved to {save_folder}/edited_{num}_{cnt}.png")
        return Response(status_code=204)
    except Exception as e:
        traceback.print_exc()
        # 失败时仍然返回 204，让客户端不感知，靠 output_path 是否存在来判断
        # 如果你想让客户端知道失败，把这里改成 raise HTTPException(500, str(e))
        return Response(status_code=204)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base", type=int, default=8000, help="基础端口号")
    args = parser.parse_args()
    print(f"Running on {args.base}")

    uvicorn.run(app, host="0.0.0.0", port=args.base, workers=1)