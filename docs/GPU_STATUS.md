# GPU/CPU Status - Current Configuration

## Current Setup Status

### ✅ Using Apple GPU (MPS):
1. **Embeddings Model** (Sentence Transformers)
   - Device: **Apple GPU (MPS)** ✓
   - Purpose: Converting text to vectors for document search
   - Speed: Fast (accelerated by GPU)

### ❌ Using CPU Only:
2. **Main LLaMA 2 Model** (CTransformers/GGML)
   - Device: **CPU only** ❌
   - Purpose: Generating AI responses
   - Speed: Slow (30-90 seconds per query)
   - Reason: **GGML format doesn't support GPU acceleration**

## Why LLaMA Model Can't Use GPU

### Technical Limitation:
- **CTransformers** uses **GGML format** 
- **GGML** is designed for **CPU inference only**
- It does **NOT support**:
  - NVIDIA CUDA (Linux/Windows)
  - Apple Metal/MPS (macOS)
  - Any GPU acceleration

### Why GGML is CPU-Only:
1. Uses CPU-optimized integer math
2. Designed for portability (works on any CPU)
3. Very efficient on CPU (better than FP32 models)
4. But cannot leverage GPU hardware

## What This Means

**Current Performance:**
- Document search: **Fast** (GPU-accelerated embeddings)
- AI response generation: **Slow** (CPU-only model)

**Total Query Time:** 30-90 seconds (mostly waiting for CPU model)

## Options to Use GPU

### Option 1: Different Model Format (Complex)
Would require:
- PyTorch model format (not GGML)
- Different model loader
- More memory
- Not currently implemented

### Option 2: Linux Server with NVIDIA GPU
- Deploy on Linux server
- Use CUDA-capable model
- 10-30x faster responses
- Requires Linux + NVIDIA GPU setup

### Option 3: Accept CPU Inference (Current)
- Use current setup
- Accept 30-90 second response times
- Works on any system
- No additional setup needed

## Verification

Check current status in the app:
- Look at sidebar "System Info" section
- Should show: "Device: Apple GPU (MPS)"
- This confirms embeddings are on GPU
- But main model will always show CPU usage

## Summary

| Component | Device | Accelerated? | Why? |
|-----------|--------|--------------|------|
| Embeddings | Apple GPU (MPS) | ✅ Yes | PyTorch supports MPS |
| LLaMA Model | CPU | ❌ No | GGML format limitation |
| Vector Search | CPU | ❌ No | FAISS-CPU is CPU-only |

**Bottom Line:** 
- Embeddings: ✅ GPU accelerated
- Main model: ❌ Must use CPU (GGML limitation)
- This is the expected behavior with current setup

