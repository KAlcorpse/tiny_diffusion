import os
from pipeline import setup_pipeline

def generate_image(prompt, negative_prompt="ugly, blurry, poor quality", steps=25):
    # Initialize our optimized pipeline
    pipe = setup_pipeline()

    print(f"Generating image for: '{prompt}'")
    print("Cooking in the latent space... this might take a minute on a 3050.")

    # Generate the image
    # We keep steps around 20-30; going higher takes longer with little benefit.
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=7.5 # How strictly the UNet should follow your prompt
    )

    image = result.images[0]

    # Save the output
    output_path = os.path.join("outputs", "generation_01.png")
    image.save(output_path)
    print(f"Success! Image saved to {output_path}")

if __name__ == "__main__":
    my_prompt = "a majestic cat wearing a glowing purple wizard hat, highly detailed, fantasy concept art, 4k"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    generate_image(prompt=my_prompt)