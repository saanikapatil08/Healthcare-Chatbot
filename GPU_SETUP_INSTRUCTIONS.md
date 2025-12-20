# GPU Setup Instructions - Full GPU Acceleration

## Important Notes

### macOS Limitation:
- **macOS does NOT support NVIDIA CUDA**
- CUDA only works on **Linux/Windows with NVIDIA GPUs**
- On macOS, we use **Apple Metal (MPS)** instead, which provides GPU acceleration

### Model Format Change:
To use GPU acceleration, we need to switch from:
- ❌ **GGML format** (CTransformers) - CPU only
- ✅ **PyTorch format** (HuggingFace Transformers) - GPU compatible

## Setup Steps

### Step 1: Install GPU Dependencies

```bash
cd /Users/saanika/UMDCodingJourney/Healthcare-Chatbot
source .venv/bin/activate

# Install GPU-accelerated libraries
pip install transformers==4.35.0
pip install accelerate==0.24.1
pip install bitsandbytes  # Optional, for quantization
```

### Step 2: Download GPU-Compatible Model

For macOS (Apple Silicon):
```bash
# Option 1: Small model for testing (fastest)
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Option 2: Medium model (better quality)
huggingface-cli download microsoft/DialoGPT-medium

# Option 3: Full LLaMA 2 (requires HuggingFace access)
# You need to request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf
```

For Linux/Windows with NVIDIA GPU:
```bash
# Download full LLaMA 2 model
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf
```

### Step 3: Update Model Configuration

In the Streamlit sidebar:
- **Model Path:** Leave empty (will use HuggingFace model name)
- Or specify: `meta-llama/Llama-2-7b-chat-hf` (for HuggingFace models)

### Step 4: Restart App

```bash
streamlit run app.py
```

The app will automatically:
1. Detect GPU (MPS on macOS, CUDA on Linux/Windows)
2. Load model on GPU
3. Run all inference on GPU

## Performance Expectations

### With GPU Acceleration:

| Platform | Device | Expected Speed |
|----------|--------|----------------|
| macOS (M1/M2) | Apple GPU (MPS) | 5-15 seconds per query |
| Linux/Windows | NVIDIA GPU (CUDA) | 1-5 seconds per query |
| Any (fallback) | CPU | 30-90 seconds per query |

## Troubleshooting

### "No GPU available"
- **macOS:** Ensure you have Apple Silicon (M1/M2/M3)
- **Linux/Windows:** Ensure NVIDIA drivers and CUDA are installed

### "Out of memory"
- Use a smaller model (TinyLlama instead of full LLaMA)
- Reduce `max_new_tokens` parameter
- Use quantization (4-bit or 8-bit)

### "Model loading failed"
- Check internet connection (models download from HuggingFace)
- Verify HuggingFace login if using gated models
- Try a smaller model first

## Current Status

The code has been updated to:
- ✅ Detect and use GPU automatically
- ✅ Fall back to CPU if GPU unavailable
- ✅ Use MPS on macOS, CUDA on Linux/Windows
- ✅ Accelerate embeddings on GPU
- ✅ Accelerate main model on GPU

## Next Steps

1. Install dependencies: `pip install transformers accelerate`
2. Choose and download a GPU-compatible model
3. Restart the app
4. Verify GPU usage in the app's system info

