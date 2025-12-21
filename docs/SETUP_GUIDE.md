# Step-by-Step Setup Guide - Medical Chatbot

Complete guide to get your medical chatbot up and running.

---

## Prerequisites Check

You're on macOS, so we'll use CPU mode (GPU support is for Linux/Windows only).

---

## STEP 1: Verify Virtual Environment is Active

```bash
# Check if you're in the virtual environment
which python

# Should show: /Users/saanika/UMDCodingJourney/Healthcare-Chatbot/.venv/bin/python
# If not, activate it:
source .venv/bin/activate
```

**What this does:** Ensures you're using the project's isolated Python environment.

---

## STEP 2: Verify Dependencies are Installed

```bash
# Check if PyTorch is installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check if Streamlit is installed
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"

# Check if LangChain is installed
python -c "import langchain; print('LangChain: OK')"
```

**Expected output:**
- PyTorch: 2.8.0 (or similar)
- Streamlit: 1.29.0
- LangChain: OK

**If anything is missing:**
```bash
pip install -r requirements.txt
```

---

## STEP 3: Download LLaMA 2 Model

You need to download the LLaMA 2 model. Choose one option:

### Option A: Using HuggingFace CLI (Recommended)

```bash
# Install huggingface-cli if not installed
pip install huggingface-hub

# Download the model
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGML \
    llama-2-7b-chat.ggmlv3.q8_0.bin \
    --local-dir models/ \
    --local-dir-use-symlinks False
```

### Option B: Manual Download

1. Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
2. Download: `llama-2-7b-chat.ggmlv3.q8_0.bin` (about 6.7 GB)
3. Place it in: `Healthcare-Chatbot/models/`

**Verify download:**
```bash
ls -lh models/llama-2-7b-chat.ggmlv3.q8_0.bin
# Should show file size ~6.7GB
```

---

## STEP 4: Prepare Medical Documents

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Add your medical PDF files to the data/ directory
# For example:
# cp ~/Documents/medical_book.pdf data/
# cp ~/Downloads/health_guide.pdf data/
```

**What files to add:**
- Medical textbooks
- Research papers
- Health guides
- Any PDF documents with medical information

**Verify files:**
```bash
ls -lh data/
# Should list your PDF files
```

**Note:** If you don't have medical PDFs yet, you can still test the setup. The chatbot will work but won't have medical context.

---

## STEP 5: Create Vector Database

This processes your PDFs and creates a searchable database:

```bash
# Run the ingestion script
python ingest.py
```

**What this does:**
- Loads all PDFs from `data/` directory
- Splits them into chunks
- Creates embeddings
- Saves vector database to `vectorstore/db_faiss/`

**Expected output:**
```
Loading documents from data/...
Loaded X documents
Splitting documents into chunks...
Created X text chunks
Initializing embedding model...
Embedding model initialized
Creating FAISS vector database...
Vector database saved to vectorstore/db_faiss

DATABASE STATISTICS
==================================================
Total Documents: X
Total Chunks: X
...
```

**If you see errors:**
- Make sure PDFs are in `data/` directory
- Check that files are valid PDFs
- Ensure enough disk space

---

## STEP 6: Verify Vector Database Created

```bash
# Check if database files exist
ls -lh vectorstore/db_faiss/

# Should show:
# - index.faiss
# - index.pkl
```

---

## STEP 7: Launch the Application

```bash
# Start Streamlit app
streamlit run app.py
```

**What happens:**
- Browser opens automatically at `http://localhost:8501`
- If not, open browser and go to: `http://localhost:8501`

**Expected:**
- You'll see the Medical AI Assistant interface
- Sidebar shows configuration options
- Main area shows welcome message

---

## STEP 8: Initialize the Chatbot

In the Streamlit interface:

1. **Check Sidebar Settings:**
   - Model Path: `models/llama-2-7b-chat.ggmlv3.q8_0.bin`
   - Vector DB Path: `vectorstore/db_faiss`
   - Adjust if needed

2. **Click "Initialize Chatbot" button**
   - Wait for model to load (may take 1-2 minutes)
   - You'll see loading messages:
     - "Loading embeddings model..."
     - "Loading LLaMA 2 model..."
     - "Loading vector database..."
     - "Setting up QA chain..."

3. **Success Message:**
   - "Chatbot initialized successfully!"

---

## STEP 9: Test the Chatbot

Once initialized:

1. **Go to "Chat" tab**
2. **Type a question:**
   - "What are the symptoms of diabetes?"
   - "How is hypertension diagnosed?"
   - "What causes headaches?"
3. **Click "Get Answer"**
4. **Wait for response** (may take 15-30 seconds on CPU)

**Expected:**
- Answer appears below
- Response time shown
- Confidence score displayed
- Sources listed (if available)

---

## STEP 10: Explore Features

### Chat Tab
- Ask medical questions
- View conversation history
- See source documents

### Analytics Tab
- View performance metrics
- See response time trends
- Check confidence scores

### About Tab
- Read project documentation
- Learn about features

---

## Troubleshooting

### Issue: "Model file not found"
**Solution:**
```bash
# Check if model file exists
ls -lh models/

# If missing, download again (see STEP 3)
```

### Issue: "Vector database not found"
**Solution:**
```bash
# Re-run ingestion
python ingest.py
```

### Issue: "Out of memory" or slow responses
**Solution:**
- This is normal on CPU mode (macOS)
- Responses take 15-30 seconds
- Consider using smaller model (q4_0 instead of q8_0)
- Close other applications to free RAM

### Issue: App won't start
**Solution:**
```bash
# Check if Streamlit is installed
pip install streamlit

# Try running again
streamlit run app.py
```

### Issue: Port 8501 already in use
**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## Quick Command Reference

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Check dependencies
python -c "import torch, streamlit, langchain; print('All OK')"

# 3. Download model (if not done)
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGML \
    llama-2-7b-chat.ggmlv3.q8_0.bin --local-dir models/

# 4. Add PDFs to data/ directory
# (Copy your PDFs manually)

# 5. Create vector database
python ingest.py

# 6. Run application
streamlit run app.py
```

---

## What Each Component Does

- **app.py**: Main Streamlit web interface
- **ingest.py**: Processes PDFs and creates vector database
- **config.py**: Configuration settings
- **gpu_config.py**: GPU settings (for future Linux deployment)
- **models/**: Contains the LLaMA 2 AI model
- **data/**: Your medical PDF documents
- **vectorstore/**: Searchable database of your documents
- **logs/**: Application logs
- **reports/**: Evaluation reports

---

## Next Steps After Setup

1. **Add more documents** to `data/` and re-run `python ingest.py`
2. **Test with various medical questions**
3. **Check Analytics tab** to see performance
4. **Review About tab** for advanced features
5. **Customize prompts** in `config.py` if needed

---

## Performance Expectations (CPU Mode)

- **First load**: 1-2 minutes
- **Response time**: 15-30 seconds per question
- **Memory usage**: 8-10 GB RAM
- **Model size**: 6.7 GB (q8_0) or 3.5 GB (q4_0)

For faster performance, deploy on Linux server with GPU (see `docs/GPU_SETUP.md`).

---

## Need Help?

1. Check logs in `logs/` directory
2. Review error messages in terminal
3. Ensure all prerequisites are met
4. Verify file paths are correct

---

**You're all set! Start with STEP 1 and work through each step.**

