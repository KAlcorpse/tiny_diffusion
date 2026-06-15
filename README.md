# tiny-diffusion: Latent Diffusion Engine for Resource-Constrained Environments

A highly optimized, hardware-aware implementation of the Stable Diffusion v1-5 inference pipeline in PyTorch. This project focuses on decoupling massive generative model graphs from physical VRAM constraints, enabling full-resolution ($512\times512$) image synthesis on entry-level consumer hardware (4GB VRAM) without compromising output mathematical accuracy.

---

## Architectural Enhancements

Standard Latent Diffusion pipelines demand a monolithic memory allocation, resulting in immediate Out-Of-Memory (OOM) failures on low-tier GPUs. This engine implements a system-level memory hierarchy rewrite to bypass the VRAM ceiling:

* **Asynchronous Model Demand-Paging (CPU Offloading):** Sub-components (Text Encoder, UNet, VAE) stay in system RAM and are paged onto the GPU only when executing, dynamically evicting inactive blocks.
* **Sequential Attention Slicing:** Splits quadratic self-attention matrices into smaller, localized sequential computations to flatten activation spikes.
* **Spatial VAE Tiling/Slicing:** Segments large latent feature maps into overlapping spatial patches during the final decoding phase to bypass structural reconstruction bottlenecks.
* **Mixed-Precision Execution:** Enforces a strict `float16` execution graph, cutting core tensor memory in half.

---

## Performance Benchmarks

### Environment Setup
* **GPU:** NVIDIA GeForce RTX 3050 (4GB VRAM)
* **Compute Backend:** PyTorch CUDA 12.x / FP16 Precision
* **Base Architecture:** RunwayML Stable Diffusion v1-5

| Pipeline Configuration | Peak VRAM Consumption | Execution Status | Result |
| :--- | :--- | :--- | :--- |
| **Baseline (FP32, Monolithic)** | > 8.4 GB | FAILED (CUDA OOM) | Instant CUDA OOM on initialization |
| **Mid-Tier (FP16 only)** | > 4.2 GB | FAILED (VAE Decode OOM) | Crashed during VAE reconstruction |
| **OptiDiff Engine (Ours)** | **1979.17 MB** | Stable / Success | **54% VRAM Footprint Reduction** |

---

## Trade-offs 
Bypassing hardware ceilings removes the need for expensive upgrades, but staging tensors over the PCIe bus introduces a compute-latency tax that increases overall generation time.
---

## Project Structure

```text
├── local_model/         # Standalone weights (.safetensors)
├── outputs/             # Runtime destination folder for generated outputs
├── src/
│   ├── benchmark.py     # Hardware profiling framework calculating VRAM performance metrics
│   └── generate.py      # Change user_prompt here to check results
├── .gitignore           # Filter protecting tracking history against large model weights
└── README.md
```
## Wanna try?

# 1. Install dependencies
pip install torch diffusers transformers accelerate omegaconf

# 2. Grab the model weights cleanly (4.27GB)
mkdir -p local_model
curl -L -o local_model/v1-5-pruned-emaonly.safetensors [https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors)

# 3. Check them numbers yourself
python src/benchmark.py

# 4. Start prompting directly in your terminal!
python src/generate.py