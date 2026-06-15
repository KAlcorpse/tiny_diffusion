import torch
import time
import gc
import os
from diffusers import StableDiffusionPipeline

PROMPT = "a majestic cat wearing a glowing purple wizard hat, highly detailed, fantasy concept art, 4k"
CHECKPOINT_PATH = "local_model/v1-5-pruned-emaonly.safetensors"

def flush_memory():
    """Clears VRAM to ensure isolated tests."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def run_test(test_name, use_fp16=False, use_slicing=False, use_offload=False):
    print(f"\n[{test_name}] Initializing...")
    flush_memory()
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Could not find weights file at {CHECKPOINT_PATH}")
        return None

    try:
        dtype = torch.float16 if use_fp16 else torch.float32
        pipe = StableDiffusionPipeline.from_single_file(
            CHECKPOINT_PATH,
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        if use_slicing:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            
        if use_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
            
        print(f"[{test_name}] Generating image...")
        start_time = time.time()
        _ = pipe(prompt=PROMPT, num_inference_steps=20).images[0]
        end_time = time.time()
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Success!")
        print(f"Time: {end_time - start_time:.2f} seconds")
        print(f"Peak VRAM: {peak_vram:.2f} MB")
        
        del pipe
        flush_memory()
        return peak_vram

    except torch.cuda.OutOfMemoryError:
        print("FAILED: CUDA Out of Memory (OOM) Crash!")
        return None

if __name__ == "__main__":
    print("=== STARTING VRAM BENCHMARK ===")
    
    # Test 1: The standard Hugging Face implementation without any optimizations (FP32, no slicing/offload)
    baseline_vram = run_test(
        "Baseline (FP32, No Opts)", 
        use_fp16=False, use_slicing=False, use_offload=False
    )
    
    # Test 2: Half-Precision only (FP16, no slicing/offload)
    fp16_vram = run_test(
        "Mid-Tier (FP16, No Slicing/Offload)", 
        use_fp16=True, use_slicing=False, use_offload=False
    )
    
    # Test 3: FP16 + Slicing + Offload (Fully optimized for 4GB VRAM)
    optimized_vram = run_test(
        "Fully Optimized (Your Setup)", 
        use_fp16=True, use_slicing=True, use_offload=True
    )

    print("\n=== BENCHMARK RESULTS ===")
    if fp16_vram and optimized_vram:
        reduction = ((fp16_vram - optimized_vram) / fp16_vram) * 100
        print(f"Optimization reduced VRAM from {fp16_vram:.2f}MB to {optimized_vram:.2f}MB")
        print(f"Total VRAM Reduction (from FP16 baseline): **{reduction:.1f}%**")
        print("\nUse this exact percentage in your resume!")
    elif optimized_vram:
        print(f"Optimized setup ran successfully at {optimized_vram:.2f} MB Peak VRAM.")
        print("Baseline runs threw OOM as expected on 4GB hardware!")
