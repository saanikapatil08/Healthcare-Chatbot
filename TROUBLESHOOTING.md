# Troubleshooting Guide - Chatbot Not Working

## Current Issues Found

### Issue 1: Low Memory (Critical)
**Status:** Your system has only **1.3GB available RAM** out of 8GB (84% used)

**Impact:** This is causing the chatbot to be extremely slow (10+ minutes per query)

**Solutions:**

#### Immediate Fix:
1. **Close other applications:**
   - Close browsers (Chrome, Safari, etc.)
   - Close other IDEs or text editors
   - Close any other Python processes
   - Check Activity Monitor and close memory-intensive apps

2. **Free up RAM:**
   ```bash
   # Check what's using memory
   top -o mem -n 10
   ```

3. **Restart your Mac** (if possible) to free up RAM

#### Better Solution - Use Smaller Model:
The Q8_0 model (6.7GB) is too large for your available RAM. Use Q4_0 (3.5GB) instead:

```bash
cd /Users/saanika/UMDCodingJourney/Healthcare-Chatbot
source .venv/bin/activate

# Download smaller model
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGML \
    llama-2-7b-chat.ggmlv3.q4_0.bin \
    --local-dir models/

# Then update Model Path in sidebar to:
# models/llama-2-7b-chat.ggmlv3.q4_0.bin
```

### Issue 2: Import Error (Handled by Patch)
The RetrievalQA import error is automatically handled by the compatibility patch in app.py. This should not prevent the app from running.

## Step-by-Step Recovery

### Step 1: Free Up Memory
```bash
# Check memory usage
top -o mem

# Kill unnecessary Python processes (if any)
pkill -f python
```

### Step 2: Restart Streamlit App
```bash
# Stop current app (Ctrl+C in terminal)
# Then restart:
cd /Users/saanika/UMDCodingJourney/Healthcare-Chatbot
source .venv/bin/activate
streamlit run app.py
```

### Step 3: Initialize Chatbot (In Browser)
- Click "Initialize Chatbot" button
- Wait for initialization (1-2 minutes)
- You should see "Chatbot initialized successfully!"

### Step 4: Test with Simple Question
- Type: "What is diabetes?"
- Click "Get Answer"
- Wait 30-90 seconds (should be faster now with more RAM)

## If Still Not Working

### Check Logs
Look at the terminal where Streamlit is running for error messages.

### Run Diagnostic
```bash
cd /Users/saanika/UMDCodingJourney/Healthcare-Chatbot
source .venv/bin/activate
python diagnose.py
```

### Common Issues

**1. "Out of memory" error:**
- Solution: Close more applications or use Q4_0 model

**2. "Model file not found":**
- Solution: Check model path in sidebar matches actual file location

**3. "Vector database not found":**
- Solution: Run `python ingest.py` to create database

**4. App crashes on initialization:**
- Solution: Check terminal for error messages, ensure 8GB+ total RAM

## Recommended Setup for Your System

Given you have 8GB RAM total:
1. **Use Q4_0 model** (3.5GB) instead of Q8_0 (6.7GB)
2. **Close other apps** before running chatbot
3. **Restart Mac** if memory is still low
4. **Consider upgrading RAM** if possible (16GB recommended)

## Performance Expectations

### With Current Setup (Low Memory):
- Initialization: 2-5 minutes
- Query time: 2-5 minutes (or timeout)
- Status: **Not recommended**

### After Freeing RAM:
- Initialization: 1-2 minutes  
- Query time: 30-90 seconds
- Status: **Works but slow**

### With Q4_0 Model + Free RAM:
- Initialization: 30-60 seconds
- Query time: 15-45 seconds
- Status: **Recommended**

## Still Having Issues?

Run the diagnostic script and share the output:
```bash
python diagnose.py
```

