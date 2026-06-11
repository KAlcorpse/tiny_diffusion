from diffusers import StableDiffusionPipeline
import torch

def setup_pipeline(model_id="runwayml/stable-diffusion-v1-5"):
    print(f"Initializing pipeline with resume support...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir="./model_cache", 
        resume_download=True       
    )

    # 4GB VRAM optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    return pipe