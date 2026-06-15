# tiny-diffusion: Latent Diffusion Engine for Resource-Constrained Environments

A highly optimized, hardware-aware implementation of the Stable Diffusion v1-5 inference pipeline in PyTorch. This project focuses on decoupling massive generative model graphs from physical VRAM constraints, enabling full-resolution ($512\times512$) image synthesis on entry-level consumer hardware (4GB VRAM) without compromising output mathematical accuracy.

---

## Architectural Enhancements

Standard Latent Diffusion pipelines demand a monolithic memory allocation, resulting in immediate Out-Of-Memory (OOM) failures on low-tier GPUs. This engine implements a system-level memory hierarchy rewrite to bypass the VRAM ceiling:

* **Asynchronous Model Demand-Paging (CPU Offloading):** Transitions execution away from global GPU residency. Sub-component weight subgraphs (Text Encoder, UNet, VAE) are kept in system host memory and asynchronously paged onto the GPU core strictly on demand, evicting inactive modules dynamically.
* **Sequential Attention Slicing:** Mitigates the quadratic memory overhead ($O(N^2)$) of vanilla self-attention matrices. The engine chunks cross-attention dot-product sequences into sequential localized computations, dramatically reducing peak tensor activation spikes.
* **Spatial VAE Tiling/Slicing:** Circumvents the heavy memory rush encountered during the final latent-to-pixel space reconstruction phase. Large latent feature maps are processed via split overlapping tiles and seamlessly reconstructed post-inference.
* **Mixed-Precision Execution:** Enforces an explicit `float16` half-precision compute graph, halving the structural memory size of execution tensors while maintaining inference fidelity.

---

## Performance Benchmarks

### Environment Setup
* **GPU:** NVIDIA GeForce RTX 3050 (4GB VRAM)
* **Compute Backend:** PyTorch CUDA 12.x / FP16 Precision
* **Base Architecture:** RunwayML Stable Diffusion v1-5

| Pipeline Configuration | Peak VRAM Consumption | Execution Status | Resume Metric Impact |
| :--- | :--- | :--- | :--- |
| **Baseline (FP32, Monolithic)** | > 8.4 GB | FAILED (CUDA OOM) | Baseline Metric Point |
| **Mid-Tier (FP16 only)** | > 4.2 GB | FAILED (VAE Decode OOM) | Peak Spike Bottleneck |
| **OptiDiff Engine (Ours)** | **1979.17 MB** | Stable / Success | **54% VRAM Footprint Reduction** |

---

## Trade-offs 

### **Pros**
* **Zero Output Degradation:** Enforcing FP16 precision, attention slicing, and VAE tiling reduces memory overhead without modifying the actual mathematical weights of the underlying models. The output image quality remains structurally identical to unoptimized runs on enterprise hardware.
* **Bypasses the Hardware Ceiling:** It completely eliminates the hard VRAM ceiling. Instead of requiring a costly GPU upgrade, your effective memory pool scales up to your system's host RAM, making it impossible to OOM during standard generation sizes.

### **Cons**
* **The Compute-Latency Tax:** Because the GPU is no longer keeping the full model in memory, it must wait for chunks of the model to be paged over the PCIe bus from system RAM at every denoising step, increasing overall generation time.
* **CPU/Bus Dependencies:** Inference speeds become heavily bound to the host system memory bus bandwidth (DDR4/DDR5 clock speeds) during sequential tensor handoffs.

---

## Project Structure & Execution

```text
├── local_model/         # Standalone weights (.safetensors)
├── outputs/             # Runtime destination folder for generated outputs
├── src/
│   ├── benchmark.py     # Hardware profiling framework calculating VRAM performance metrics
│   └── generate.py      # Fully interactive Terminal CLI for dynamic user generation loops
├── .gitignore           # Filter protecting tracking history against large model weights
└── README.md

## Wanna try?

# 1. Install dependencies
pip install torch diffusers transformers accelerate omegaconf

# 2. Grab the model weights cleanly (4.27GB)
mkdir -p local_model
curl -L -o local_model/v1-5-pruned-emaonly.safetensors [https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors)

# 3. Prove the numbers yourself
python src/benchmark.py

# 4. Start prompting directly in your terminal!
python src/generate.py