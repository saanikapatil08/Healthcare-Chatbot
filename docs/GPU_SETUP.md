# GPU Setup Guide - Medical Chatbot

Complete guide to setup and optimize the medical chatbot with GPU acceleration.

---

## Table of Contents

1. [GPU Requirements](#gpu-requirements)
2. [Installation](#installation)
3. [GPU Configuration](#gpu-configuration)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting](#troubleshooting)
6. [Benchmarks](#benchmarks)

---

## GPU Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: Version 11.8 or 12.1
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space

### Recommended Specifications
- **GPU**: RTX 3080/4070 or better (12GB+ VRAM)
- **CUDA**: Latest version (12.1+)
- **RAM**: 32GB system RAM
- **Storage**: 100GB SSD

### Optimal Setup
- **GPU**: RTX 4090 or A100 (24GB+ VRAM)
- **CUDA**: 12.1+
- **RAM**: 64GB system RAM
- **Storage**: 200GB NVMe SSD

---

## Installation

### Step 1: Check GPU Compatibility

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 2: Install CUDA Toolkit

**For Ubuntu/Linux:**
```bash
# Download CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Install
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**For Windows:**
1. Download CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Run the installer
3. Add CUDA to system PATH

### Step 3: Install cuDNN

```bash
# Download cuDNN from NVIDIA (requires registration)
# Extract and copy files

# Linux
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Step 4: Create Virtual Environment

```bash
# Create environment
python -m venv venv_gpu

# Activate (Linux/Mac)
source venv_gpu/bin/activate

# Activate (Windows)
venv_gpu\Scripts\activate
```

### Step 5: Install GPU Dependencies

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install FAISS-GPU
pip install faiss-gpu==1.7.2
```

### Step 6: Verify Installation

```python
# test_gpu.py
import torch
import faiss

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"FAISS GPU support: {hasattr(faiss, 'StandardGpuResources')}")

# Test GPU
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    print(f"Test tensor on GPU: {x.device}")
    print("GPU is working!")
```

Run the test:
```bash
python test_gpu.py
```

---

## GPU Configuration

### Option 1: Use GPU Presets (Recommended)

The application includes optimized presets for common GPUs:

```python
from gpu_config import GPUConfig, GPUPresets

# Auto-detect and optimize
GPUConfig.setup_gpu()
gpu_info = GPUConfig.get_gpu_info()
GPUConfig.optimize_for_gpu_memory(gpu_info['total_memory_gb'])

# Or use specific preset
GPUPresets.rtx_3080()  # For RTX 3080
GPUPresets.rtx_4090()  # For RTX 4090
GPUPresets.a100()      # For A100
```

### Option 2: Manual Configuration

Edit `gpu_config.py`:

```python
class GPUConfig:
    # Basic GPU settings
    USE_GPU = True
    GPU_DEVICE = 0
    
    # Memory settings
    GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory
    
    # Model settings
    GPU_LAYERS = 43  # All layers on GPU
    CONTEXT_LENGTH = 4096
    MAX_NEW_TOKENS = 512
    BATCH_SIZE = 8
    
    # Optimization
    USE_FP16 = True  # Half precision
    USE_FLASH_ATTENTION = True
```

### GPU Preset Configurations

| GPU Model | VRAM | GPU Layers | Context Length | Batch Size | Expected Speed |
|-----------|------|------------|----------------|------------|----------------|
| RTX 3060 | 12GB | 32 | 2048 | 4 | 2-4 tokens/s |
| RTX 3080 | 10-12GB | 38 | 3072 | 6 | 5-8 tokens/s |
| RTX 4070 | 12GB | 40 | 3072 | 8 | 8-12 tokens/s |
| RTX 4090 | 24GB | 43 | 8192 | 16 | 20-30 tokens/s |
| A100 | 40-80GB | 43 | 16384 | 32 | 40-60 tokens/s |

---

## Performance Optimization

### 1. Enable Mixed Precision

```python
# In gpu_config.py
USE_FP16 = True  # Use FP16 (2x faster, half memory)

# For Ampere GPUs (RTX 30xx/40xx, A100)
USE_BF16 = True  # Better numerical stability
```

### 2. Optimize Batch Size

```python
# Find optimal batch size
def find_optimal_batch_size(model, max_batch=32):
    for batch_size in range(1, max_batch + 1):
        try:
            # Test with batch_size
            dummy_input = torch.randn(batch_size, 512).cuda()
            model(dummy_input)
            print(f"Batch size {batch_size}: OK")
        except RuntimeError:  # OOM
            print(f"Optimal batch size: {batch_size - 1}")
            return batch_size - 1
```

### 3. Enable Flash Attention

```bash
# Install Flash Attention (for RTX 30xx+, A100)
pip install flash-attn --no-build-isolation
```

```python
# Enable in config
USE_FLASH_ATTENTION = True
ENABLE_MEMORY_EFFICIENT_ATTENTION = True
```

### 4. Use Model Quantization

```python
# 4-bit quantization (4x smaller, slightly lower quality)
MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"

# 8-bit quantization (2x smaller, minimal quality loss)
MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"
```

### 5. Optimize FAISS Index

```python
# Use IVF index for large datasets
import faiss

dimension = 384  # embedding dimension
nlist = 100  # number of clusters

quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Move to GPU
gpu_resources = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
```

### 6. Enable Gradient Checkpointing

```python
# Saves memory at cost of slight speed decrease
GRADIENT_CHECKPOINTING = True
```

### 7. Multi-GPU Setup (Optional)

```python
# Distribute across multiple GPUs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use GPUs 0-3

# In model loading
from torch.nn import DataParallel
model = DataParallel(model)
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce GPU layers:**
```python
GPUConfig.GPU_LAYERS = 32  # Instead of 43
```

2. **Decrease batch size:**
```python
GPUConfig.BATCH_SIZE = 4  # Instead of 8
```

3. **Reduce context length:**
```python
GPUConfig.CONTEXT_LENGTH = 2048  # Instead of 4096
```

4. **Enable CPU offloading:**
```python
GPUConfig.OFFLOAD_TO_CPU = True
```

5. **Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

### Issue 2: Slow Performance

**Solutions:**

1. **Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

2. **Enable FP16:**
```python
GPUConfig.USE_FP16 = True
```

3. **Increase batch size:**
```python
GPUConfig.BATCH_SIZE = 16
```

4. **Use faster model:**
```python
MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"  # 4-bit
```

### Issue 3: FAISS GPU Not Working

**Solutions:**

1. **Check FAISS GPU installation:**
```python
import faiss
print(hasattr(faiss, 'StandardGpuResources'))
```

2. **Reinstall FAISS-GPU:**
```bash
pip uninstall faiss-cpu faiss-gpu
pip install faiss-gpu==1.7.2
```

3. **Use CPU fallback:**
```python
GPUConfig.USE_FAISS_GPU = False
```

### Issue 4: Driver Issues

**Solutions:**

1. **Update NVIDIA drivers:**
```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-535

# Check version
nvidia-smi
```

2. **Check CUDA compatibility:**
```bash
nvidia-smi  # Check CUDA version
nvcc --version  # Check compiler version
```

### Issue 5: Model Not Loading

**Solutions:**

1. **Verify model file:**
```bash
ls -lh models/
file models/llama-2-7b-chat.ggmlv3.q8_0.bin
```

2. **Check file permissions:**
```bash
chmod 644 models/*.bin
```

3. **Try CPU first:**
```python
GPUConfig.USE_GPU = False
GPUConfig.GPU_LAYERS = 0
```

---

## Benchmarks

### Performance Comparison

| Configuration | Response Time | Memory Usage | Tokens/sec |
|--------------|---------------|--------------|------------|
| **CPU Only** | 15-30s | 8GB RAM | 1-2 |
| **GPU (RTX 3060)** | 3-6s | 8GB VRAM | 5-8 |
| **GPU (RTX 3080)** | 2-4s | 10GB VRAM | 10-15 |
| **GPU (RTX 4090)** | 1-2s | 12GB VRAM | 25-35 |
| **GPU (A100)** | 0.5-1s | 16GB VRAM | 50-70 |

### Memory Usage by Configuration

| Model Quantization | Model Size | GPU Memory | CPU Memory |
|-------------------|------------|------------|------------|
| Q4_0 (4-bit) | 3.5GB | 4-6GB | 2GB |
| Q5_0 (5-bit) | 4.3GB | 5-7GB | 2GB |
| Q8_0 (8-bit) | 6.7GB | 7-10GB | 2GB |
| FP16 | 13GB | 14-18GB | 3GB |

### Optimization Impact

| Optimization | Speed Gain | Memory Savings |
|-------------|------------|----------------|
| FP16 | 2x | 50% |
| Flash Attention | 1.5x | 30% |
| Batch Processing | 3-5x | - |
| FAISS-GPU | 10-20x | - |
| 4-bit Quantization | - | 70% |

---

## Quick Start Commands

### Start with GPU (Automatic):
```bash
python app.py
# GPU will be auto-detected and configured
```

### Start with Specific Preset:
```python
# In app.py or Python console
from gpu_config import GPUPresets

GPUPresets.rtx_4090()  # Use RTX 4090 preset
```

### Monitor GPU Usage:
```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Run app
streamlit run app.py
```

### Clear GPU Memory:
```python
from gpu_config import GPUConfig
GPUConfig.clear_gpu_cache()
```

---

## Additional Resources

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [FAISS GPU Documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
- [Hugging Face Optimization](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [LLaMA 2 Performance Guide](https://github.com/facebookresearch/llama)

---

## Success Checklist

- [ ] NVIDIA drivers installed and updated
- [ ] CUDA toolkit installed (11.8 or 12.1)
- [ ] PyTorch with CUDA support installed
- [ ] FAISS-GPU installed and working
- [ ] GPU detected by PyTorch
- [ ] Model loads successfully on GPU
- [ ] Embeddings running on GPU
- [ ] Vector database on GPU (optional)
- [ ] Performance benchmarks meet expectations
- [ ] Memory usage within limits

---

**Your medical chatbot is now GPU-accelerated!**

