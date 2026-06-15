import os
import torch
from diffusers import StableDiffusionPipeline

def setup_pipeline():
    """Initializes the StableDiffusionPipeline using a standalone local checkpoint 
    and applies critical memory-optimizations for 4GB VRAM hardware."""
    checkpoint_path = "local_model/v1-5-pruned-emaonly.safetensors"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ Missing weights file at {checkpoint_path}. "
            f"Please run the curl download step first!"
        )

    print(f"Initializing pipeline from standalone checkpoint: {checkpoint_path}")
    
    # Load directly from our explicit single file checkpoint
    pipe = StableDiffusionPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # Core System-ML optimizations to fit a 4GB+ model onto a 4GB RTX 3050 GPU
    print("Applying memory-managed architecture optimizations...")
    pipe.enable_attention_slicing()  # Breaks down quadratic attention matrices
    pipe.enable_vae_slicing()        # Limits peak memory spikes during image reconstruction
    pipe.enable_model_cpu_offload()  # Sequentially pages model sub-blocks into VRAM on demand

    return pipe

def generate_image(prompt, negative_prompt="ugly, blurry, poor quality", steps=25):
    """Executes the inference loop and saves the final output image."""
    # Initialize our hardware-aware optimized pipeline
    pipe = setup_pipeline()

    print(f"\nGenerating image for: '{prompt}'")
    print("Cooking in the latent space... execution may take up to a minute on your hardware.")

    # Run execution loop through the UNet denoiser
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=7.5  # Controls prompt adherence weight
    )

    image = result.images[0]

    # Target path for final verification
    output_path = os.path.join("outputs", "generational.png")
    image.save(output_path)
    print(f" Success! High-fidelity image saved safely to: {output_path}")

if __name__ == "__main__":
    my_prompt = "a devil wearing a tuxedo, highly detailed, fantasy concept art, 4k"
    
    # Create target directory for generation output files
    os.makedirs("outputs", exist_ok=True)
    
    generate_image(prompt=my_prompt)