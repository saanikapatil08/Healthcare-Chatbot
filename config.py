"""
Configuration file for Medical Chatbot
Centralized settings for easy customization
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for Medical Chatbot"""
    
    # ============================================================================
    # MODEL SETTINGS
    # ============================================================================
    
    # LLaMA 2 Model Configuration
    MODEL_PATH: str = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"
    MODEL_TYPE: str = "llama"
    
    # Model Parameters
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.2
    CONTEXT_LENGTH: int = 2048
    REPETITION_PENALTY: float = 1.1
    TOP_K: int = 50
    TOP_P: float = 0.9
    GPU_LAYERS: int = 0  # Set to > 0 for GPU acceleration
    
    # ============================================================================
    # EMBEDDING SETTINGS
    # ============================================================================
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # Change to "cuda" for GPU
    NORMALIZE_EMBEDDINGS: bool = True
    
    # Alternative embedding models (uncomment to use):
    # EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Better quality
    # EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # Faster
    
    # ============================================================================
    # VECTOR DATABASE SETTINGS
    # ============================================================================
    
    # FAISS Database
    VECTOR_DB_PATH: str = "vectorstore/db_faiss"
    
    # Document Processing
    DATA_PATH: str = "data"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Retrieval Settings
    RETRIEVAL_K: int = 3  # Number of documents to retrieve
    SEARCH_TYPE: str = "similarity"  # or "mmr" for Maximum Marginal Relevance
    
    # ============================================================================
    # PROMPT ENGINEERING
    # ============================================================================
    
    # System Prompt Template
    SYSTEM_PROMPT: str = """You are a knowledgeable and professional medical assistant. 
Your goal is to provide accurate, helpful medical information based on the provided context.

Guidelines:
1. Provide accurate information based solely on the context provided
2. Use clear, professional medical terminology
3. If information is not in the context, clearly state that
4. Always recommend consulting healthcare professionals for diagnosis and treatment
5. Be empathetic and supportive in your responses
6. Include relevant details and explanations when appropriate
"""
    
    # Question-Answer Prompt Template
    QA_PROMPT_TEMPLATE: str = """Use the following context to answer the question accurately and professionally.

Context: {context}

Question: {question}

Instructions:
1. Provide accurate medical information based on the context
2. If the information is not in the context, clearly state that
3. Use clear, professional medical terminology
4. Include relevant details and explanations
5. Always recommend consulting healthcare professionals for diagnosis

Answer: """
    
    # ============================================================================
    # UI CONFIGURATION
    # ============================================================================
    
    # Streamlit Page Config
    PAGE_TITLE: str = "Medical AI Assistant - LLaMA 2"
    PAGE_ICON: str = None
    LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"
    
    # Theme Colors
    PRIMARY_COLOR: str = "#1f77b4"
    SECONDARY_COLOR: str = "#2ca02c"
    BACKGROUND_COLOR: str = "#f0f2f6"
    
    # ============================================================================
    # PERFORMANCE SETTINGS
    # ============================================================================
    
    # Caching
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600  # seconds
    
    # Metrics
    ENABLE_METRICS: bool = True
    METRICS_SAVE_PATH: str = "logs/metrics.json"
    
    # Logging
    ENABLE_LOGGING: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/chatbot.log"
    
    # ============================================================================
    # ADVANCED SETTINGS
    # ============================================================================
    
    # Text Splitting Strategy
    TEXT_SEPARATORS: list = ["\n\n", "\n", " ", ""]
    LENGTH_FUNCTION = len
    
    # Retrieval Strategy
    RETRIEVAL_SEARCH_KWARGS: Dict[str, Any] = {
        'k': 3,
        'fetch_k': 10,  # For MMR
        'lambda_mult': 0.5  # For MMR diversity
    }
    
    # Chain Settings
    CHAIN_TYPE: str = "stuff"  # or "map_reduce", "refine", "map_rerank"
    RETURN_SOURCE_DOCUMENTS: bool = True
    VERBOSE: bool = False
    
    # ============================================================================
    # SAFETY & COMPLIANCE
    # ============================================================================
    
    # Disclaimer Text
    DISCLAIMER: str = """
    **Medical Disclaimer**
    
    This AI assistant provides information for educational purposes only. 
    It does not provide medical advice, diagnosis, or treatment. 
    
    Always consult qualified healthcare professionals for:
    - Medical diagnosis
    - Treatment decisions
    - Medication advice
    - Emergency situations
    
    If you have a medical emergency, call emergency services immediately.
    """
    
    # Content Filtering
    ENABLE_CONTENT_FILTER: bool = True
    BLOCKED_KEYWORDS: list = [
        "prescription",
        "prescribe",
        "diagnose with",
        "self-medicate"
    ]
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            'model': cls.MODEL_PATH,
            'model_type': cls.MODEL_TYPE,
            'max_new_tokens': cls.MAX_NEW_TOKENS,
            'temperature': cls.TEMPERATURE,
            'context_length': cls.CONTEXT_LENGTH,
            'repetition_penalty': cls.REPETITION_PENALTY,
            'top_k': cls.TOP_K,
            'top_p': cls.TOP_P,
            'gpu_layers': cls.GPU_LAYERS,
        }
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding configuration dictionary"""
        return {
            'model_name': cls.EMBEDDING_MODEL_NAME,
            'model_kwargs': {'device': cls.EMBEDDING_DEVICE},
            'encode_kwargs': {'normalize_embeddings': cls.NORMALIZE_EMBEDDINGS}
        }
    
    @classmethod
    def get_retrieval_config(cls) -> Dict[str, Any]:
        """Get retrieval configuration dictionary"""
        return {
            'search_type': cls.SEARCH_TYPE,
            'search_kwargs': cls.RETRIEVAL_SEARCH_KWARGS
        }
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_PATH,
            Path(cls.VECTOR_DB_PATH).parent,
            Path(cls.MODEL_PATH).parent,
            Path(cls.METRICS_SAVE_PATH).parent if cls.ENABLE_METRICS else None,
            Path(cls.LOG_FILE).parent if cls.ENABLE_LOGGING else None,
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check model file exists
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"Model file not found: {cls.MODEL_PATH}")
        
        # Check data directory exists
        if not Path(cls.DATA_PATH).exists():
            errors.append(f"Data directory not found: {cls.DATA_PATH}")
        
        # Check vector DB exists
        if not Path(cls.VECTOR_DB_PATH).exists():
            errors.append(f"Vector database not found: {cls.VECTOR_DB_PATH}")
        
        # Validate parameters
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 2:
            errors.append("Temperature must be between 0 and 2")
        
        if cls.MAX_NEW_TOKENS < 1:
            errors.append("MAX_NEW_TOKENS must be positive")
        
        if cls.RETRIEVAL_K < 1:
            errors.append("RETRIEVAL_K must be positive")
        
        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  {error}")
            return False
        
        print("Configuration validated successfully")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        print(f"\nMODEL SETTINGS:")
        print(f"  Model Path: {cls.MODEL_PATH}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print(f"  Max Tokens: {cls.MAX_NEW_TOKENS}")
        print(f"  Context Length: {cls.CONTEXT_LENGTH}")
        
        print(f"\nEMBEDDING SETTINGS:")
        print(f"  Model: {cls.EMBEDDING_MODEL_NAME}")
        print(f"  Device: {cls.EMBEDDING_DEVICE}")
        
        print(f"\nDATABASE SETTINGS:")
        print(f"  Vector DB: {cls.VECTOR_DB_PATH}")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"  Retrieval K: {cls.RETRIEVAL_K}")
        
        print(f"\nUI SETTINGS:")
        print(f"  Page Title: {cls.PAGE_TITLE}")
        print(f"  Layout: {cls.LAYOUT}")
        
        print(f"\nPERFORMANCE:")
        print(f"  Caching: {'Enabled' if cls.ENABLE_CACHING else 'Disabled'}")
        print(f"  Metrics: {'Enabled' if cls.ENABLE_METRICS else 'Disabled'}")
        print(f"  Logging: {'Enabled' if cls.ENABLE_LOGGING else 'Disabled'}")
        print("="*60 + "\n")


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class FastConfig(Config):
    """Optimized for speed"""
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.3
    CHUNK_SIZE = 300
    RETRIEVAL_K = 2


class AccurateConfig(Config):
    """Optimized for accuracy"""
    MAX_NEW_TOKENS = 768
    TEMPERATURE = 0.1
    CHUNK_SIZE = 800
    RETRIEVAL_K = 5
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class BalancedConfig(Config):
    """Balanced configuration (default)"""
    pass


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Print default configuration
    Config.print_config()
    
    # Validate configuration
    Config.validate_config()
    
    # Create necessary directories
    Config.ensure_directories()
    
    print("\nExample usage in your code:")
    print("""
from config import Config

# Use configuration
model_config = Config.get_model_config()
embedding_config = Config.get_embedding_config()

# Access settings
print(f"Model path: {Config.MODEL_PATH}")
print(f"Temperature: {Config.TEMPERATURE}")
    """)

