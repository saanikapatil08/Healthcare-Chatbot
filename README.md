# Healthcare Chatbot — GenAI-Powered Medical Information Assistant

## Overview
The Healthcare Chatbot is a GenAI-powered medical information assistant built using a Retrieval-Augmented Generation (RAG) architecture. It combines large language models with semantic document retrieval to generate accurate, context-aware, and grounded responses from trusted medical literature.

This project demonstrates strong foundations in GenAI system design, vector search, prompt engineering, and local model deployment.

---

## Problem Statement
Medical information is often:
- Buried inside large and unstructured documents
- Difficult to search contextually
- Prone to hallucinations when queried using standalone LLMs

This project addresses these challenges by grounding LLM responses in verified medical documents using semantic search.

---

## Key Features

### Core Capabilities
- **LLaMA 2 Integration**: Quantized 7B model optimized for CPU inference
- **RAG Architecture**: Combines retrieval with generation for accurate responses
- **FAISS Vector Database**: Lightning-fast semantic search
- **Advanced Prompt Engineering**: Medical-domain optimized prompts
- **Real-time Performance Metrics**: Track response times and confidence scores
- **Source Attribution**: Every answer includes document references
- **Interactive Streamlit UI**: User-friendly interface with visualizations
  
---

### Performance Monitoring
- Response time tracking
- Confidence score analysis
- Query history and analytics
- Visual performance dashboards
  
---

## Solution
The system follows a Retrieval-Augmented Generation pipeline:
1. Medical documents are ingested and chunked
2. Chunks are converted into embeddings
3. Embeddings are stored in a vector database
4. Relevant context is retrieved at query time
5. An LLM generates a grounded response using retrieved context

This approach improves factual accuracy and transparency.

---

## Technology Stack
- Language Model: LLaMA 2 (GGML format)
- Embeddings: Sentence Transformers
- Vector Database: FAISS
- Frameworks: Python, LangChain
- UI: Streamlit
- Deployment: Local CPU / GPU (configurable)

---

## Project Structure

```
Healthcare-Chatbot/
├── .venv/ # Python virtual environment (local)
│
├── data/ # Medical reference documents (not tracked)
│ ├── The_GALE_ENCYCLOPEDIA_of_Medicine_.pdf
│ └── The-Gale-Encyclopedia-of-Medicine-.pdf
│
├── docs/ # Project documentation
│ ├── CPU_VS_GPU_EXPLANATION.md
│ ├── deployment_guide.md
│ ├── GPU_IMPLEMENTATION.md
│ ├── GPU_SETUP_INSTRUCTIONS.md
│ ├── GPU_SETUP.md
│ ├── GPU_STATUS.md
│ ├── PROJECT_DOCUMENTATION.md
│ ├── QUICK_GPU_SETUP.md
│ ├── QUICK_START.md
│ ├── SETUP_GUIDE.md
│ └── TROUBLESHOOTING.md
│
├── models/ # LLaMA model binaries (ignored by git)
│ ├── llama-2-7b-chat.ggmlv3.q4_0.bin
│ └── llama-2-7b-chat.ggmlv3.q8_0.bin
│
├──app.py # Main Streamlit chatbot application
├── config.py # Central configuration file
├── diagnose.py # System diagnostics and health checks
├── gpu_config.py # GPU detection and configuration
├── gpu_monitor.py # Runtime GPU monitoring
├── ingest.py # Document ingestion & vectorization pipeline
├── test_evaluation.py # Evaluation and performance testing
│
├── vectorstore/ # dbstore (ignored by git)
│ └── db_faiss/ # FAISS vector database (generated)
│ ├── index.faiss
│ └── index.pkl
│
├── .gitignore # Git ignore rules
├── README.md # Project overview and quick start
├── requirements.txt # Python dependencies
└── RUN_APP.sh # Shell script to launch the application
```

## Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **RAM**: Minimum 16GB
- **OS**: Windows, Linux, or macOS
- **CPU**: Modern multi-core processor (GPU optional)

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download LLaMA 2 Model**

Download the quantized LLaMA 2 model from HuggingFace:

```bash
# Option 1: Using huggingface-cli
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGML \
    llama-2-7b-chat.ggmlv3.q8_0.bin \
    --local-dir models/
```

Or manually download from:
- [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)

5. **Prepare Medical Documents**

Place your medical PDF documents in the `data/` directory:

```bash
mkdir -p data
# Add your PDF files to data/
```

6. **Create Vector Database**

```bash
python ingest.py
```

This will:
- Load all PDFs from `data/` directory
- Split documents into optimized chunks
- Generate embeddings using Sentence Transformers
- Create and save FAISS vector database

7. **Launch the Application**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### First Time Setup

1. **Configure Settings**: In the sidebar, verify model and database paths
2. **Initialize Chatbot**: Click "Initialize Chatbot" button
3. **Start Asking**: Type your medical question and click "Get Answer"

### Sample Questions

```
"What are the common symptoms of diabetes?"
"How is hypertension diagnosed?"
"What are the treatment options for migraine?"
"Explain the difference between Type 1 and Type 2 diabetes"
"What are the risk factors for heart disease?"
```

### Features Overview

#### Chat Tab
- Ask medical questions
- View detailed answers with sources
- Track conversation history
- See real-time metrics (response time, confidence)

#### Analytics Tab
- Performance metrics dashboard
- Response time trends
- Confidence score distribution
- Detailed query history

#### About Tab
- Project documentation
- Technology stack details
- Setup instructions
- Performance optimization tips

## Configuration

### Model Configuration

Edit in `app.py` or Streamlit sidebar:

```python
# Model parameters
MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
CONTEXT_LENGTH = 2048
```

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2`

Alternative options:
- `sentence-transformers/all-mpnet-base-v2` (Better quality, slower)
- `sentence-transformers/paraphrase-MiniLM-L6-v2` (Faster, smaller)

### Retrieval Settings

```python
# Number of relevant documents to retrieve
RETRIEVAL_K = 3

# Chunk size for document splitting
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

## Advanced Features

### Custom Prompt Engineering

Modify the prompt template in `app.py`:

```python
def create_prompt_template(self) -> PromptTemplate:
    template = """You are a medical expert...
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
```

### Adding New Documents

1. Add PDFs to `data/` directory
2. Run ingestion script:
```bash
python ingest.py
```
3. Restart the application

### Performance Optimization

**For Faster Responses:**
- Use smaller chunk sizes (300-400 characters)
- Reduce retrieval K to 2
- Increase temperature slightly (0.3-0.4)

**For Better Accuracy:**
- Increase chunk size (600-800 characters)
- Increase retrieval K to 4-5
- Lower temperature (0.1-0.2)

## Performance Metrics

### Expected Performance

| Metric | Value |
|--------|-------|
| Average Response Time | 5-15 seconds (CPU) |
| Average Response Time | 1-3 seconds (GPU) |
| Memory Usage | 8-10 GB |
| Confidence Score | 70-90% |

### Benchmarks

Tested on:
- **CPU**: Intel i7-10700K @ 3.8GHz
- **RAM**: 32GB DDR4
- **Documents**: 100 medical PDFs (~2000 pages)
- **Chunks**: 15,000 text segments

## Security & Privacy

**Important Disclaimers**

1. **Not for Clinical Use**: This chatbot is for informational purposes only
2. **No Medical Advice**: Always consult healthcare professionals
3. **Data Privacy**: Keep sensitive medical data secure
4. **Local Deployment**: All processing happens locally (no cloud APIs)

## Troubleshooting

### Common Issues

**Issue: Model not loading**
```
Solution: Check model file path and ensure file is downloaded correctly
```

**Issue: Out of memory**
```
Solution: Close other applications or use a machine with more RAM
```

**Issue: Slow responses**
```
Solution: 
- Reduce max_new_tokens to 256
- Reduce context_length to 1024
- Use GPU if available
```

**Issue: Vector DB not found**
```
Solution: Run ingest.py to create the database
```

### Performance Tips

1. **Use SSD**: Store vector DB on SSD for faster loading
2. **Optimize Chunks**: Experiment with chunk sizes
3. **Batch Processing**: Process multiple documents together
4. **GPU Acceleration**: Use FAISS-GPU for faster search

## Updates & Maintenance

### Updating Documents

```bash
# Add new PDFs to data/
# Re-run ingestion
python ingest.py

# Restart application
streamlit run app.py
```

### Model Updates

To use a different model:
1. Download new model to `models/`
2. Update `MODEL_PATH` in configuration
3. Restart application

## Future Enhancements

- Multi-modal support (images, lab reports)
- Fine-tuning on medical datasets
- Integration with medical APIs
- Multi-language support
- Voice input/output
- Export chat history
- Integration with medical APIs
- Mobile app version
- Docker containerization

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## References

### Papers & Research
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

### Libraries & Tools
- [LangChain](https://python.langchain.com/)
- [FAISS](https://faiss.ai/)
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [HuggingFace](https://huggingface.co/)

### Medical Resources
- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/)
- [Medical Textbooks](https://www.ncbi.nlm.nih.gov/books/)

## Contact

- **Project Maintainer**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

## Acknowledgments

- Based on the original [Llama2-Medical-Chatbot](https://github.com/AIAnytime/Llama2-Medical-Chatbot) by AIAnytime
- LLaMA 2 by Meta AI
- Streamlit team for the amazing framework
- HuggingFace for model hosting

---

**Disclaimer**: This tool is for educational and research purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

**Built for Healthcare AI**

