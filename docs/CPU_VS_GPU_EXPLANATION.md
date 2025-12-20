# CPU vs GPU - Why Your Chatbot Runs on CPU

## Why CPU Mode on macOS?

### 1. **macOS Doesn't Support NVIDIA CUDA**
- macOS doesn't have NVIDIA GPU drivers
- CUDA (NVIDIA's GPU computing platform) doesn't work on macOS
- Only Linux and Windows support NVIDIA CUDA for GPU acceleration

### 2. **Your Hardware: Apple Silicon (M1/M2/M3)**
- You have Apple Silicon (ARM64) - this is great!
- Apple Silicon has a Neural Engine and GPU
- However, these use **Metal** (Apple's GPU framework), not CUDA

### 3. **LLaMA 2 Model Format (GGML)**
- The LLaMA 2 model uses **GGML format** (via CTransformers)
- GGML is optimized for **CPU inference**
- It doesn't support GPU acceleration (neither CUDA nor Metal)
- This is why the main model runs on CPU

### 4. **What CAN Use GPU on macOS?**
- **Embeddings** (Sentence Transformers) - CAN use Metal/MPS
- **Vector Search** (FAISS) - CPU only
- **LLaMA Model** (CTransformers/GGML) - CPU only

## Current Configuration

| Component | Device | Why? |
|-----------|--------|------|
| LLaMA 2 Model | CPU | GGML format is CPU-only |
| Embeddings | CPU/MPS* | Can use Metal if configured |
| Vector Search | CPU | FAISS-CPU is CPU-only |
| Text Processing | CPU | General processing |

*We've now enabled MPS for embeddings to speed up document processing!

## Performance Expectations

### CPU Mode (Current Setup)
- **Response Time**: 15-30 seconds per query
- **Model Loading**: 30-60 seconds
- **Memory**: 8-10 GB RAM

### If You Had NVIDIA GPU (Linux/Windows)
- **Response Time**: 2-5 seconds per query
- **Model Loading**: 5-10 seconds
- **Memory**: 10-12 GB VRAM

## Why GGML is CPU-Only

1. **GGML Format**: Designed specifically for CPU optimization
2. **Quantization**: Uses CPU-friendly integer math
3. **Portability**: Works on any CPU without GPU drivers
4. **Efficiency**: Very efficient on CPU (better than running FP32 models)

## Alternatives for GPU Acceleration

### Option 1: Use PyTorch-based Models (Supports Metal)
- Would require different model format
- More complex setup
- Not available in current architecture

### Option 2: Deploy on Linux Server with NVIDIA GPU
- Much faster (10-30x speedup)
- Requires Linux server with NVIDIA GPU
- See `docs/GPU_SETUP.md` for details

### Option 3: Use Smaller Model
- Q4_0 model is smaller and faster
- Still CPU-only but faster inference
- Trade-off: slightly lower quality

## Summary

**Your chatbot runs on CPU because:**
1. macOS doesn't support NVIDIA CUDA
2. LLaMA 2 GGML format is optimized for CPU
3. This is the standard setup for CPU inference
4. We've enabled MPS for embeddings to help speed up document processing

**The warning you see** is harmless - just PyTorch checking for CUDA classes.

**Performance is normal** - 15-30 seconds per query is expected for CPU inference with this model size.

