# Deployment Guide - Medical AI Chatbot

Complete guide for deploying your medical chatbot to production

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Production Deployment Options](#production-deployment-options)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Performance Optimization](#performance-optimization)
6. [Security Considerations](#security-considerations)
7. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Development Setup

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download LLaMA 2 model
# Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
# Download: llama-2-7b-chat.ggmlv3.q8_0.bin
# Place in: models/

# 5. Prepare data
mkdir -p data
# Add your medical PDFs to data/ directory

# 6. Create vector database
python ingest.py

# 7. Run application
streamlit run app.py
```

### Development Best Practices

```python
# Use configuration file
from config import Config

# Enable debug mode
Config.VERBOSE = True
Config.LOG_LEVEL = "DEBUG"

# Test with small dataset first
Config.CHUNK_SIZE = 300
Config.RETRIEVAL_K = 2
```

---

## Production Deployment Options

### Option 1: Streamlit Cloud (Recommended for Prototypes)

**Pros:**
- Free tier available
- Easy deployment
- Automatic HTTPS
- Built-in authentication

**Cons:**
- Resource limitations
- Not suitable for large models

**Steps:**

1. Push code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository
4. Configure secrets in dashboard
5. Deploy!

**Note:** Streamlit Cloud has memory limits. Consider smaller models.

---

### Option 2: AWS Deployment

#### AWS EC2 Deployment

**Recommended Instance:** `r6i.2xlarge` (64GB RAM)

```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type r6i.2xlarge \
  --key-name your-key

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx

# 4. Clone and setup
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Setup systemd service
sudo nano /etc/systemd/system/medical-chatbot.service
```

**Service Configuration:**

```ini
[Unit]
Description=Medical Chatbot Streamlit App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/medical-chatbot
Environment="PATH=/home/ubuntu/medical-chatbot/venv/bin"
ExecStart=/home/ubuntu/medical-chatbot/venv/bin/streamlit run app.py --server.port 8501

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable medical-chatbot
sudo systemctl start medical-chatbot
```

**Nginx Configuration:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

### Option 3: Google Cloud Platform

**Using Google Compute Engine:**

```bash
# Create instance
gcloud compute instances create medical-chatbot \
  --machine-type=n2-highmem-4 \
  --zone=us-central1-a \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB

# SSH and setup (similar to AWS)
```

**Using Cloud Run (Containerized):**

```bash
# Build and push Docker image
gcloud builds submit --tag gcr.io/your-project/medical-chatbot

# Deploy
gcloud run deploy medical-chatbot \
  --image gcr.io/your-project/medical-chatbot \
  --platform managed \
  --memory 16Gi \
  --cpu 4
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p data models vectorstore logs

# Download model (or mount as volume)
# RUN curl -L "MODEL_URL" -o models/llama-2-7b-chat.ggmlv3.q8_0.bin

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  medical-chatbot:
    build: .
    container_name: medical-chatbot
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./vectorstore:/app/vectorstore
      - ./logs:/app/logs
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
    mem_limit: 16g
    cpus: 4

  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - medical-chatbot
    restart: unless-stopped
```

**Build and Run:**

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Performance Optimization

### 1. Model Optimization

```python
# Use quantized models
MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"  # 4-bit quantization

# Optimize generation parameters
Config.MAX_NEW_TOKENS = 256  # Reduce for faster responses
Config.TEMPERATURE = 0.2     # Lower for more focused answers
```

### 2. Caching Strategy

```python
import streamlit as st
from functools import lru_cache

@st.cache_resource
def load_chatbot():
    """Cache chatbot initialization"""
    chatbot = MedicalChatbot(model_path, db_path)
    chatbot.initialize()
    return chatbot

@lru_cache(maxsize=100)
def cached_query(question: str):
    """Cache frequent queries"""
    return chatbot.query(question)
```

### 3. Database Optimization

```python
# Optimize FAISS index
import faiss

# Use IVF (Inverted File) for large datasets
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Use GPU if available
index = faiss.index_cpu_to_gpu(res, 0, index)
```

### 4. Load Balancing

```nginx
upstream medical_chatbot {
    least_conn;
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    location / {
        proxy_pass http://medical_chatbot;
    }
}
```

---

## Security Considerations

### 1. Authentication

```python
# Add authentication to Streamlit
import streamlit_authenticator as stauth

# Create authentication object
authenticator = stauth.Authenticate(
    credentials,
    'medical_chatbot',
    'secure_key',
    cookie_expiry_days=30
)

# Add login widget
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show chatbot
    main()
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

### 2. Environment Variables

```bash
# .env file
MODEL_PATH=/path/to/model
DB_PATH=/path/to/db
SECRET_KEY=your-secret-key
API_KEY=your-api-key
```

```python
# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
```

### 3. HTTPS Configuration

```bash
# Generate SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 4. Rate Limiting

```python
from datetime import datetime, timedelta

# Simple rate limiter
rate_limit_store = {}

def check_rate_limit(user_id: str, limit: int = 10, window: int = 60):
    """Check if user exceeded rate limit"""
    now = datetime.now()
    
    if user_id not in rate_limit_store:
        rate_limit_store[user_id] = []
    
    # Remove old requests
    rate_limit_store[user_id] = [
        ts for ts in rate_limit_store[user_id]
        if now - ts < timedelta(seconds=window)
    ]
    
    # Check limit
    if len(rate_limit_store[user_id]) >= limit:
        return False
    
    rate_limit_store[user_id].append(now)
    return True
```

---

## Monitoring & Maintenance

### 1. Logging Setup

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chatbot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log queries
logger.info(f"Query: {question}")
logger.info(f"Response time: {response_time}s")
```

### 2. Performance Monitoring

```python
import psutil
import time

def monitor_system():
    """Monitor system resources"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent
    }
```

### 3. Health Checks

```python
# Add health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        if chatbot.llm is None:
            return {'status': 'unhealthy', 'reason': 'Model not loaded'}
        
        # Check vector DB
        if chatbot.vectorstore is None:
            return {'status': 'unhealthy', 'reason': 'Vector DB not loaded'}
        
        return {'status': 'healthy'}
    except Exception as e:
        return {'status': 'unhealthy', 'reason': str(e)}
```

### 4. Backup Strategy

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup vector database
tar -czf "$BACKUP_DIR/vectorstore_$DATE.tar.gz" vectorstore/

# Backup logs
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" logs/

# Keep only last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

---

## Scaling Strategies

### Horizontal Scaling

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: medical-chatbot
  template:
    metadata:
      labels:
        app: medical-chatbot
    spec:
      containers:
      - name: chatbot
        image: medical-chatbot:latest
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
          limits:
            memory: "20Gi"
            cpu: "6"
```

### Database Replication

```python
# Read replicas for vector database
class VectorStorePool:
    def __init__(self, db_paths: List[str]):
        self.stores = [
            FAISS.load_local(path, embeddings)
            for path in db_paths
        ]
        self.current_index = 0
    
    def get_store(self):
        """Round-robin load balancing"""
        store = self.stores[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.stores)
        return store
```

---

## Troubleshooting Guide

### Common Issues

**Issue 1: Out of Memory**
```bash
# Solution: Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Issue 2: Slow Response Times**
```python
# Solution: Reduce model complexity
Config.MAX_NEW_TOKENS = 256
Config.CONTEXT_LENGTH = 1024
```

**Issue 3: Vector DB Corruption**
```bash
# Solution: Rebuild database
python ingest.py --rebuild
```

---

## Pre-Deployment Checklist

- [ ] Test all features locally
- [ ] Run evaluation suite
- [ ] Configure environment variables
- [ ] Setup SSL certificates
- [ ] Configure backup strategy
- [ ] Setup monitoring and alerts
- [ ] Test with production data
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Documentation updated

---

## Support & Resources

- **Documentation**: `/docs`
- **Issues**: GitHub Issues
- **Community**: Discord/Slack
- **Email**: support@example.com

---

**Last Updated**: December 2024
**Version**: 1.0.0

