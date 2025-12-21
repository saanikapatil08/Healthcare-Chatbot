# Quick GPU Setup Guide

## What Changed

✅ **Code Updated:** The app now automatically uses GPU when available
- **macOS:** Uses Apple GPU (MPS) 
- **Linux/Windows:** Uses NVIDIA GPU (CUDA)
- **Fallback:** Uses CPU if GPU not available

## Setup (2 Steps)

### Step 1: Install Dependencies
```bash
cd /Users/saanika/UMDCodingJourney/Healthcare-Chatbot
source .venv/bin/activate
pip install transformers accelerate
```

### Step 2: Restart App
```bash
streamlit run app.py
```

The app will now:
1. ✅ Automatically detect GPU
2. ✅ Load model on GPU (MPS or CUDA)
3. ✅ Run all inference on GPU
4. ✅ Show GPU status in sidebar

## Model Selection

**For macOS (Apple Silicon):**
- Uses: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (small, fast, GPU-accelerated)
- Downloads automatically on first use

**For Linux/Windows (NVIDIA GPU):**
- Uses: `meta-llama/Llama-2-7b-chat-hf` (full model, requires HuggingFace access)
- See GPU_SETUP_INSTRUCTIONS.md for setup

## Expected Performance

| Platform | Device | Speed |
|----------|--------|-------|
| macOS | Apple GPU (MPS) | 5-15 seconds |
| Linux/Windows | NVIDIA GPU (CUDA) | 1-5 seconds |
| Any (fallback) | CPU | 30-90 seconds |

## Verification

After starting the app, check the sidebar:
- Should show: "Using Apple GPU (MPS) for model acceleration" (macOS)
- Or: "Using NVIDIA GPU (CUDA)" (Linux/Windows)

## Troubleshooting

**"GPU loading failed":**
- Check if transformers is installed: `pip install transformers`
- Verify GPU is available: Python code will auto-detect
- Falls back to CPU automatically

**"Model download failed":**
- Check internet connection
- Model downloads automatically from HuggingFace

**Still slow?**
- Verify GPU is being used (check sidebar message)
- Check system memory (need 4GB+ free)

