"""
GPU-Optimized Configuration for Medical Chatbot
Leverages GPU for faster inference and better memory management
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any


class GPUConfig:
    """GPU-Optimized Configuration"""
    
    # ============================================================================
    # GPU SETTINGS
    # ============================================================================
    
    # GPU Configuration
    USE_GPU: bool = True
    GPU_DEVICE: int = 0  # GPU device ID (0 for first GPU)
    CUDA_VISIBLE_DEVICES: str = "0"  # Set which GPUs to use
    
    # Check GPU availability
    DEVICE = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
    
    # Memory Management
    GPU_MEMORY_FRACTION: float = 0.8  # Use 80% of GPU memory
    ENABLE_MEMORY_EFFICIENT_ATTENTION: bool = True
    USE_FLASH_ATTENTION: bool = True  # If available
    
    # Mixed Precision Training/Inference
    USE_FP16: bool = True  # Use half precision (faster, less memory)
    USE_BF16: bool = False  # Use bfloat16 (better for modern GPUs like A100)
    
    # ============================================================================
    # MODEL SETTINGS (GPU-Optimized)
    # ============================================================================
    
    # Model Configuration
    MODEL_PATH: str = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"  # 4-bit quantization for GPU
    MODEL_TYPE: str = "llama"
    
    # GPU-specific parameters
    GPU_LAYERS: int = 43  # Number of layers to offload to GPU (all 43 for 7B model)
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.2
    CONTEXT_LENGTH: int = 4096  # Increased for GPU
    BATCH_SIZE: int = 8  # Process multiple queries simultaneously
    
    # Advanced GPU settings
    N_THREADS: int = 4  # CPU threads for non-GPU operations
    N_BATCH: int = 512  # Batch size for prompt processing
    
    # Model parameters
    MODEL_CONFIG: Dict[str, Any] = {
        'context_length': CONTEXT_LENGTH,
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'repetition_penalty': 1.1,
        'top_k': 50,
        'top_p': 0.9,
        'gpu_layers': GPU_LAYERS,
        'n_threads': N_THREADS,
        'n_batch': N_BATCH,
        'f16_kv': USE_FP16,  # Use FP16 for key/value cache
    }
    
    # ============================================================================
    # EMBEDDING SETTINGS (GPU-Optimized)
    # ============================================================================
    
    # Use GPU for embeddings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = DEVICE
    EMBEDDING_BATCH_SIZE: int = 128  # Larger batch for GPU
    NORMALIZE_EMBEDDINGS: bool = True
    
    # Alternative faster models for GPU
    # EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Better quality
    # EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"  # State-of-the-art
    
    EMBEDDING_CONFIG: Dict[str, Any] = {
        'model_name': EMBEDDING_MODEL_NAME,
        'model_kwargs': {
            'device': EMBEDDING_DEVICE,
            'trust_remote_code': True
        },
        'encode_kwargs': {
            'normalize_embeddings': NORMALIZE_EMBEDDINGS,
            'batch_size': EMBEDDING_BATCH_SIZE,
            'show_progress_bar': True
        }
    }
    
    # ============================================================================
    # VECTOR DATABASE SETTINGS (GPU-Optimized)
    # ============================================================================
    
    # Use FAISS-GPU for faster similarity search
    USE_FAISS_GPU: bool = True
    FAISS_GPU_RESOURCES: int = 0  # GPU resource ID
    
    VECTOR_DB_PATH: str = "vectorstore/db_faiss"
    DATA_PATH: str = "data"
    
    # Optimized chunking for GPU processing
    CHUNK_SIZE: int = 512  # Slightly larger chunks
    CHUNK_OVERLAP: int = 64
    
    # Retrieval settings (GPU allows more documents)
    RETRIEVAL_K: int = 5  # Retrieve more documents (GPU can handle it)
    SEARCH_TYPE: str = "similarity"
    
    # ============================================================================
    # PERFORMANCE OPTIMIZATION
    # ============================================================================
    
    # Caching
    ENABLE_CACHING: bool = True
    CACHE_DIR: str = ".cache"
    
    # Parallel processing
    NUM_WORKERS: int = 4  # For data loading
    PREFETCH_FACTOR: int = 2
    
    # Memory optimization
    GRADIENT_CHECKPOINTING: bool = True
    OFFLOAD_TO_CPU: bool = False  # Keep everything on GPU if possible
    
    # ============================================================================
    # PATHS
    # ============================================================================
    
    LOG_DIR: str = "logs"
    METRICS_DIR: str = "reports"
    
    # ============================================================================
    # UI SETTINGS
    # ============================================================================
    
    PAGE_TITLE: str = "Medical AI Assistant - GPU Accelerated"
    PAGE_ICON: str = None
    LAYOUT: str = "wide"
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @classmethod
    def setup_gpu(cls):
        """Setup GPU environment"""
        if cls.USE_GPU and torch.cuda.is_available():
            # Set CUDA device
            os.environ["CUDA_VISIBLE_DEVICES"] = cls.CUDA_VISIBLE_DEVICES
            torch.cuda.set_device(cls.GPU_DEVICE)
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(
                cls.GPU_MEMORY_FRACTION, 
                cls.GPU_DEVICE
            )
            
            # Enable TF32 for better performance on Ampere GPUs
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            print(f"GPU Setup Complete")
            print(f"   Device: {torch.cuda.get_device_name(cls.GPU_DEVICE)}")
            print(f"   Memory: {torch.cuda.get_device_properties(cls.GPU_DEVICE).total_memory / 1e9:.2f} GB")
            print(f"   Compute Capability: {torch.cuda.get_device_capability()}")
        else:
            print("GPU not available, using CPU")
            cls.DEVICE = "cpu"
            cls.EMBEDDING_DEVICE = "cpu"
            cls.GPU_LAYERS = 0
            cls.USE_FAISS_GPU = False
    
    @classmethod
    def get_gpu_info(cls) -> Dict[str, Any]:
        """Get GPU information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(cls.GPU_DEVICE),
            "total_memory_gb": torch.cuda.get_device_properties(cls.GPU_DEVICE).total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(cls.GPU_DEVICE) / 1e9,
            "cached_memory_gb": torch.cuda.memory_reserved(cls.GPU_DEVICE) / 1e9,
            "compute_capability": torch.cuda.get_device_capability(cls.GPU_DEVICE),
        }
    
    @classmethod
    def clear_gpu_cache(cls):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU cache cleared")
    
    @classmethod
    def print_gpu_stats(cls):
        """Print GPU statistics"""
        info = cls.get_gpu_info()
        
        if not info["available"]:
            print("GPU not available")
            return
        
        print("\n" + "="*60)
        print("GPU STATISTICS")
        print("="*60)
        print(f"Device Name: {info['device_name']}")
        print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"Allocated Memory: {info['allocated_memory_gb']:.2f} GB")
        print(f"Cached Memory: {info['cached_memory_gb']:.2f} GB")
        print(f"Free Memory: {info['total_memory_gb'] - info['allocated_memory_gb']:.2f} GB")
        print(f"Compute Capability: {info['compute_capability']}")
        print("="*60 + "\n")
    
    @classmethod
    def optimize_for_gpu_memory(cls, available_memory_gb: float):
        """Adjust settings based on available GPU memory"""
        if available_memory_gb < 8:
            # Low memory (e.g., RTX 3060, GTX 1080)
            print("Low GPU memory detected, using conservative settings")
            cls.CONTEXT_LENGTH = 2048
            cls.MAX_NEW_TOKENS = 256
            cls.BATCH_SIZE = 4
            cls.EMBEDDING_BATCH_SIZE = 64
            cls.GPU_LAYERS = 32  # Offload some layers
            cls.RETRIEVAL_K = 3
            
        elif available_memory_gb < 16:
            # Medium memory (e.g., RTX 3080, RTX 4070)
            print("Medium GPU memory detected, using balanced settings")
            cls.CONTEXT_LENGTH = 4096
            cls.MAX_NEW_TOKENS = 512
            cls.BATCH_SIZE = 8
            cls.EMBEDDING_BATCH_SIZE = 128
            cls.GPU_LAYERS = 43
            cls.RETRIEVAL_K = 5
            
        else:
            # High memory (e.g., RTX 4090, A100)
            print("High GPU memory detected, using maximum performance settings")
            cls.CONTEXT_LENGTH = 8192
            cls.MAX_NEW_TOKENS = 1024
            cls.BATCH_SIZE = 16
            cls.EMBEDDING_BATCH_SIZE = 256
            cls.GPU_LAYERS = 43
            cls.RETRIEVAL_K = 7
    
    @classmethod
    def validate_gpu_config(cls) -> bool:
        """Validate GPU configuration"""
        errors = []
        
        if cls.USE_GPU and not torch.cuda.is_available():
            errors.append("GPU requested but CUDA not available")
        
        if cls.GPU_LAYERS > 43:
            errors.append("GPU_LAYERS cannot exceed 43 for LLaMA 2 7B")
        
        if cls.BATCH_SIZE < 1:
            errors.append("BATCH_SIZE must be positive")
        
        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("GPU Configuration validated")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current GPU configuration"""
        print("\n" + "="*60)
        print("GPU-OPTIMIZED CONFIGURATION")
        print("="*60)
        
        print(f"\nGPU SETTINGS:")
        print(f"  Use GPU: {cls.USE_GPU}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  GPU Layers: {cls.GPU_LAYERS}")
        print(f"  FP16: {cls.USE_FP16}")
        
        print(f"\nMODEL SETTINGS:")
        print(f"  Model Path: {cls.MODEL_PATH}")
        print(f"  Context Length: {cls.CONTEXT_LENGTH}")
        print(f"  Max Tokens: {cls.MAX_NEW_TOKENS}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        
        print(f"\nEMBEDDING SETTINGS:")
        print(f"  Model: {cls.EMBEDDING_MODEL_NAME}")
        print(f"  Device: {cls.EMBEDDING_DEVICE}")
        print(f"  Batch Size: {cls.EMBEDDING_BATCH_SIZE}")
        
        print(f"\nVECTOR DB SETTINGS:")
        print(f"  Use FAISS-GPU: {cls.USE_FAISS_GPU}")
        print(f"  Retrieval K: {cls.RETRIEVAL_K}")
        
        if torch.cuda.is_available():
            cls.print_gpu_stats()
        
        print("="*60 + "\n")


# ============================================================================
# GPU OPTIMIZATION PRESETS
# ============================================================================

class GPUPresets:
    """Predefined GPU configurations for different hardware"""
    
    @staticmethod
    def rtx_3060():
        """Optimized for RTX 3060 (12GB)"""
        GPUConfig.CONTEXT_LENGTH = 2048
        GPUConfig.MAX_NEW_TOKENS = 256
        GPUConfig.BATCH_SIZE = 4
        GPUConfig.GPU_LAYERS = 32
        GPUConfig.RETRIEVAL_K = 3
        print("Loaded RTX 3060 preset")
    
    @staticmethod
    def rtx_3080():
        """Optimized for RTX 3080 (10-12GB)"""
        GPUConfig.CONTEXT_LENGTH = 3072
        GPUConfig.MAX_NEW_TOKENS = 384
        GPUConfig.BATCH_SIZE = 6
        GPUConfig.GPU_LAYERS = 38
        GPUConfig.RETRIEVAL_K = 4
        print("Loaded RTX 3080 preset")
    
    @staticmethod
    def rtx_4090():
        """Optimized for RTX 4090 (24GB)"""
        GPUConfig.CONTEXT_LENGTH = 8192
        GPUConfig.MAX_NEW_TOKENS = 1024
        GPUConfig.BATCH_SIZE = 16
        GPUConfig.GPU_LAYERS = 43
        GPUConfig.RETRIEVAL_K = 7
        print("Loaded RTX 4090 preset")
    
    @staticmethod
    def a100():
        """Optimized for A100 (40-80GB)"""
        GPUConfig.CONTEXT_LENGTH = 16384
        GPUConfig.MAX_NEW_TOKENS = 2048
        GPUConfig.BATCH_SIZE = 32
        GPUConfig.GPU_LAYERS = 43
        GPUConfig.RETRIEVAL_K = 10
        GPUConfig.USE_BF16 = True
        print("Loaded A100 preset")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Setup GPU
    GPUConfig.setup_gpu()
    
    # Auto-optimize based on available memory
    if torch.cuda.is_available():
        gpu_info = GPUConfig.get_gpu_info()
        GPUConfig.optimize_for_gpu_memory(gpu_info['total_memory_gb'])
    
    # Validate configuration
    GPUConfig.validate_gpu_config()
    
    # Print configuration
    GPUConfig.print_config()
    
    print("\nExample usage:")
    print("""
# In your app.py:
from gpu_config import GPUConfig

# Setup GPU
GPUConfig.setup_gpu()

# Use GPU configuration
model_config = GPUConfig.MODEL_CONFIG
embedding_config = GPUConfig.EMBEDDING_CONFIG

# Monitor GPU
GPUConfig.print_gpu_stats()

# Clear cache when needed
GPUConfig.clear_gpu_cache()
    """)
    
    print("\nGPU Presets available:")
    print("  - GPUPresets.rtx_3060()")
    print("  - GPUPresets.rtx_3080()")
    print("  - GPUPresets.rtx_4090()")
    print("  - GPUPresets.a100()")

