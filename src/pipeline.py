from diffusers import StableDiffusionPipeline
import torch
import os

def setup_pipeline():
    # Construct the path to your actual downloaded .safetensors file
    # Based on your image, it is located inside the local_model/.cache directory tree
    base_dir = "local_model/.cache/huggingface/download"
    
    # We find the specific file matching our weights
    target_file = "v1-5-pruned-emaonly.safetensors"
    checkpoint_path = None
    
    for root, dirs, files in os.walk(base_dir):
        if target_file in files:
            checkpoint_path = os.path.join(root, target_file)
            break
            
    if not checkpoint_path:
        # Fallback if the walk fails to find it dynamically
        raise FileNotFoundError(f"Could not find {target_file} inside {base_dir}. Verify your download completed successfully!")

    print(f"Initializing pipeline from standalone checkpoint: {checkpoint_path}")
    
    # Use from_single_file to properly load a standalone .safetensors file
    pipe = StableDiffusionPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # Your 4GB VRAM optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    return pipe