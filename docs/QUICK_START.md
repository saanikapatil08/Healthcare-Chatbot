# Quick Start Guide - Running Your Medical Chatbot

## Step-by-Step Instructions

### Step 1: Navigate to Project Directory
```bash
cd /Users/saanika/UMDCodingJourney/Healthcare-Chatbot
```

### Step 2: Activate Virtual Environment
```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 3: Start the Streamlit Application
```bash
streamlit run app.py
```

### Step 4: Open in Browser
The terminal will show:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://10.0.0.86:8501
```

**Click on the URL or copy `http://localhost:8501` into your browser.**

### Step 5: Initialize the Chatbot (In Browser)
1. In the **left sidebar**, verify:
   - **Model Path**: `models/llama-2-7b-chat.ggmlv3.q8_0.bin`
   - **Vector DB Path**: `vectorstore/db_faiss`

2. Click the **"Initialize Chatbot"** button

3. Wait 1-2 minutes for:
   - Embeddings model to load
   - LLaMA 2 model to load (this takes the longest)
   - Vector database to load
   - QA chain to be set up

4. You'll see a success message when ready

### Step 6: Ask Questions
1. In the main area, type your question in the text box
   - Example: "What are the symptoms of Alzheimer's disease?"
   - Example: "How is diabetes diagnosed?"

2. Click **"Get Answer"** button

3. **Wait 30-90 seconds** (normal for CPU inference)
   - You'll see progress messages
   - Don't refresh the page!
   - The response will appear when ready

### Step 7: View Results
- Your question and answer will appear below
- You'll see:
  - Response time
  - Confidence score
  - Source documents used

## Complete Command Sequence

```bash
# Terminal 1: Start the app
cd /Users/saanika/UMDCodingJourney/Healthcare-Chatbot
source .venv/bin/activate
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Troubleshooting

### App won't start?
- Check virtual environment is activated: `which python` should show `.venv/bin/python`
- Check dependencies: `pip list | grep streamlit`

### Model not loading?
- Verify model file exists: `ls -lh models/llama-2-7b-chat.ggmlv3.q8_0.bin`
- Check you have at least 8GB free RAM

### No response after clicking "Get Answer"?
- **This is normal!** Wait 30-90 seconds on CPU
- Check the browser console for errors (F12)
- Make sure you clicked "Initialize Chatbot" first

### Vector database not found?
- Run: `python ingest.py` to create it
- Make sure you have PDFs in the `data/` folder

## Stopping the App

In the terminal where Streamlit is running:
- Press `Ctrl + C` to stop the server
- Or close the terminal window

## Next Steps

Once working:
- Try different medical questions
- Check the "Analytics" tab for performance metrics
- Read the "About" tab for more information

For faster responses, consider downloading the Q4_0 model (see sidebar instructions).

