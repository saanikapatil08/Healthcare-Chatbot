#!/usr/bin/env python3
"""
Diagnostic script to check if the chatbot setup is correct
"""

import sys
from pathlib import Path

print("="*60)
print("Medical Chatbot - Diagnostic Check")
print("="*60)

# Check 1: Python version
print("\n[1] Checking Python version...")
print(f"    Python: {sys.version}")
if sys.version_info < (3, 8):
    print("    ERROR: Python 3.8+ required")
    sys.exit(1)
else:
    print("    OK")

# Check 2: Required packages
print("\n[2] Checking required packages...")
required_packages = [
    'streamlit', 'langchain', 'ctransformers', 
    'sentence_transformers', 'faiss', 'torch', 'psutil'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f"    {package}: OK")
    except ImportError:
        print(f"    {package}: MISSING")
        missing.append(package)

if missing:
    print(f"\n    ERROR: Missing packages: {', '.join(missing)}")
    print("    Run: pip install -r requirements.txt")
    sys.exit(1)

# Check 3: Model file
print("\n[3] Checking model file...")
model_path = Path("models/llama-2-7b-chat.ggmlv3.q8_0.bin")
if model_path.exists():
    size_gb = model_path.stat().st_size / (1024**3)
    print(f"    Model found: {model_path}")
    print(f"    Size: {size_gb:.2f} GB")
    if size_gb < 5:
        print("    WARNING: Model file seems too small (< 5GB)")
    else:
        print("    OK")
else:
    print(f"    ERROR: Model not found at {model_path}")
    print("    Download from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML")

# Check 4: Vector database
print("\n[4] Checking vector database...")
db_path = Path("vectorstore/db_faiss")
if db_path.exists():
    files = list(db_path.glob("*"))
    if files:
        print(f"    Database found: {db_path}")
        print(f"    Files: {len(files)}")
        for f in files:
            size_mb = f.stat().st_size / (1024**2)
            print(f"      - {f.name}: {size_mb:.1f} MB")
        print("    OK")
    else:
        print(f"    ERROR: Database directory exists but is empty")
        print("    Run: python ingest.py")
else:
    print(f"    ERROR: Vector database not found at {db_path}")
    print("    Run: python ingest.py")

# Check 5: Data directory
print("\n[5] Checking data directory...")
data_path = Path("data")
if data_path.exists():
    pdfs = list(data_path.glob("*.pdf"))
    print(f"    Data directory: {data_path}")
    print(f"    PDF files: {len(pdfs)}")
    for pdf in pdfs[:3]:  # Show first 3
        size_mb = pdf.stat().st_size / (1024**2)
        print(f"      - {pdf.name}: {size_mb:.1f} MB")
    if len(pdfs) > 3:
        print(f"      ... and {len(pdfs) - 3} more")
    if len(pdfs) == 0:
        print("    WARNING: No PDF files found in data/")
    else:
        print("    OK")
else:
    print(f"    WARNING: Data directory not found: {data_path}")

# Check 6: Memory
print("\n[6] Checking system memory...")
try:
    import psutil
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    used_percent = memory.percent
    
    print(f"    Total RAM: {total_gb:.1f} GB")
    print(f"    Available: {available_gb:.1f} GB")
    print(f"    Used: {used_percent:.1f}%")
    
    if available_gb < 4:
        print("    WARNING: Low available memory (< 4GB). May cause slowdowns.")
    elif available_gb < 2:
        print("    ERROR: Very low memory (< 2GB). Close other applications.")
    else:
        print("    OK")
except Exception as e:
    print(f"    Could not check memory: {e}")

# Check 7: Test imports
print("\n[7] Testing critical imports...")
try:
    from langchain.llms import CTransformers
    print("    CTransformers: OK")
except Exception as e:
    print(f"    CTransformers: ERROR - {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("    HuggingFaceEmbeddings: OK")
except Exception as e:
    print(f"    HuggingFaceEmbeddings: ERROR - {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("    FAISS: OK")
except Exception as e:
    print(f"    FAISS: ERROR - {e}")

try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
    print("    RetrievalQA: OK")
except Exception as e:
    print(f"    RetrievalQA: ERROR - {e}")

print("\n" + "="*60)
print("Diagnostic Complete!")
print("="*60)
print("\nIf all checks passed, try running:")
print("  streamlit run app.py")
print("\nIf there are errors, fix them before running the app.")

