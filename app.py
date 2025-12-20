"""
Enhanced Medical Chatbot with Streamlit Interface
Features: LLaMA 2, RAG, Vector DB, Performance Metrics, Advanced Prompt Engineering
"""

# Fix langchain version compatibility issues BEFORE any other imports
import sys
import types
import warnings
import os

# Suppress all harmless warnings before any imports
# Must be done before importing any libraries that generate warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
warnings.filterwarnings('ignore', message='.*LibreSSL.*')
warnings.filterwarnings('ignore', message='.*NotOpenSSLWarning.*')
warnings.filterwarnings('ignore', message='.*Examining the path.*')

# Set environment variables to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['URLLIB3_DISABLE_WARNINGS'] = '1'

def _patch_langchain_compatibility():
    """Patch langchain_community to fix version compatibility issues"""
    # Patch memgraph_graph
    try:
        import langchain_community.graphs.memgraph_graph as memgraph_module
        if not hasattr(memgraph_module, 'RAW_SCHEMA_QUERY'):
            memgraph_module.RAW_SCHEMA_QUERY = "CALL db.schema()"
    except:
        pass
    
    # Patch baichuan chat model
    try:
        import langchain_community.chat_models.baichuan as baichuan_module
        if not hasattr(baichuan_module, '_signature'):
            baichuan_module._signature = lambda x: x
    except:
        pass

# Apply patches immediately
_patch_langchain_compatibility()

# Now import streamlit and other libraries
import streamlit as st
import time
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
import plotly.express as px

# LangChain & LLM imports (patches already applied above)
# Try PyTorch-based GPU acceleration first, fallback to CTransformers
GPU_AVAILABLE = False
try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    GPU_AVAILABLE = True
except ImportError:
    pass

# Always import CTransformers as fallback
try:
    from langchain.llms import CTransformers
except ImportError:
    CTransformers = None

from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import RetrievalQA
try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
except (ImportError, AttributeError) as e:
    try:
        from langchain.chains import RetrievalQA
    except Exception as import_error:
        _retrieval_qa_error = str(import_error)
        RetrievalQA = None
        print(f"Warning: Could not import RetrievalQA: {import_error}")

# Try to import from langchain_community for compatibility
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
except ImportError:
    # Fallback to old langchain imports
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import PyPDFLoader, DirectoryLoader

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Memory - import only if needed (can cause compatibility issues)
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    ConversationBufferMemory = None  # Optional component


@dataclass
class QueryMetrics:
    """Store metrics for each query"""
    query: str
    response: str
    response_time: float
    tokens_used: int
    sources: List[str]
    timestamp: str
    confidence_score: float


class MedicalChatbot:
    """Enhanced Medical Chatbot with RAG and Performance Tracking"""
    
    def __init__(self, model_path: str, db_path: str):
        self.model_path = model_path
        self.db_path = db_path
        self.llm = None
        self.qa_chain = None
        self.embeddings = None
        self.vectorstore = None
        self.metrics_history: List[QueryMetrics] = []
        
    def load_embeddings(self):
        """Load HuggingFace embeddings model with GPU acceleration if available"""
        import torch
        
        # Check if MPS (Apple Silicon GPU) is available
        device = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        
        if device == 'mps':
            st.info("Using Apple Silicon GPU (MPS) for embeddings acceleration")
        else:
            st.info("Using CPU for embeddings")
        
    def load_llm(self):
        """Load LLaMA 2 model with GPU acceleration (MPS for macOS, CUDA for Linux/Windows)"""
        import torch
        
        # Check for GPU availability
        use_mps = False
        use_cuda = False
        
        if torch.backends.mps.is_available():
            device = "mps"
            use_mps = True
            st.info("Using Apple GPU (MPS) for model acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            use_cuda = True
            st.info(f"Using NVIDIA GPU (CUDA) for model acceleration - Device: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            st.warning("No GPU available - falling back to CPU mode")
        
        # Try to load PyTorch model with GPU acceleration
        try:
            if GPU_AVAILABLE and (use_mps or use_cuda):
                # Use HuggingFace Transformers with GPU acceleration
                if use_mps:
                    st.info("Loading model optimized for Apple Silicon GPU (MPS)...")
                    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for MPS testing
                    torch_dtype = torch.float16
                else:  # use_cuda
                    st.info("Loading model for NVIDIA GPU (CUDA)...")
                    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Full model for CUDA
                    torch_dtype = torch.float16
                
                # Load tokenizer and model (common for both MPS and CUDA)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Load model (using dtype instead of deprecated torch_dtype)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch_dtype,  # Changed from torch_dtype to dtype
                    low_cpu_mem_usage=True
                )
                
                # Move to GPU
                if use_mps:
                    model = model.to("mps")
                elif use_cuda:
                    model = model.to("cuda")
                
                # Create pipeline - handle device mapping properly
                if use_cuda:
                    pipeline_device = 0  # CUDA device 0
                elif use_mps:
                    pipeline_device = None  # MPS handled by model.to("mps")
                else:
                    pipeline_device = -1  # CPU
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=pipeline_device,
                    max_new_tokens=256,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
                st.success(f"Model loaded on {device.upper()} (GPU accelerated)")
                return
            else:
                raise ImportError("HuggingFace transformers not available")
                
        except Exception as e:
            st.warning(f"GPU loading failed: {str(e)}. Falling back to CPU with CTransformers...")
            # Fallback to CPU CTransformers
            if CTransformers is None:
                st.error("CTransformers not available! Cannot load model.")
                self.llm = None
                return
            
            try:
                import multiprocessing
                num_threads = multiprocessing.cpu_count()
                
                self.llm = CTransformers(
                    model=self.model_path,
                    model_type="llama",
                    max_new_tokens=150,
                    temperature=0.2,
                    context_length=384,
                    gpu_layers=0,
                    threads=num_threads,
                    batch_size=1,
                    config={
                        'context_length': 384,
                        'max_new_tokens': 150,
                        'repetition_penalty': 1.1,
                        'temperature': 0.2,
                        'top_k': 30,
                        'top_p': 0.85,
                        'stream': False,
                        'threads': num_threads,
                    }
                )
                st.info("Using CPU mode (CTransformers)")
            except Exception as cpu_error:
                st.error(f"CPU loading also failed: {str(cpu_error)}")
                self.llm = None
                raise
        
    def load_vectorstore(self):
        """Load or create FAISS vector database"""
        if Path(self.db_path).exists():
            try:
                # Try with allow_dangerous_deserialization (newer versions)
                self.vectorstore = FAISS.load_local(
                    self.db_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except TypeError:
                # Fallback for older versions that don't support this parameter
                self.vectorstore = FAISS.load_local(
                    self.db_path, 
                    self.embeddings
                )
        else:
            st.error(f"Vector database not found at {self.db_path}")
            return False
        return True
    
    def create_prompt_template(self) -> PromptTemplate:
        """Create advanced prompt template with medical context"""
        template = """You are a knowledgeable medical assistant. Use the following context to answer the question accurately and professionally.

Context: {context}

Question: {question}

Instructions:
1. Provide accurate medical information based on the context
2. If the information is not in the context, clearly state that
3. Use clear, professional medical terminology
4. Include relevant details and explanations
5. Always recommend consulting healthcare professionals for diagnosis

Answer: """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def setup_qa_chain(self):
        """Setup Retrieval QA chain with custom prompt"""
        prompt = self.create_prompt_template()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={'k': 2}  # Reduced to 2 for faster retrieval
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            with st.spinner("Loading embeddings model..."):
                self.load_embeddings()
            
            with st.spinner("Loading LLaMA 2 model..."):
                self.load_llm()
                # Check if LLM was loaded successfully
                if self.llm is None:
                    st.error("Failed to load LLM model. Check error messages above.")
                    return False
            
            with st.spinner("Loading vector database..."):
                if not self.load_vectorstore():
                    return False
            
            with st.spinner("Setting up QA chain..."):
                self.setup_qa_chain()
                # Check if QA chain was created successfully
                if self.qa_chain is None:
                    st.error("Failed to setup QA chain. Check error messages above.")
                    return False
            
            return True
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query and track metrics"""
        start_time = time.time()
        
        try:
            # Execute query
            result = self.qa_chain({"query": question})
            
            # Calculate metrics
            response_time = time.time() - start_time
            answer = result['result']
            source_docs = result.get('source_documents', [])
            
            # Extract sources
            sources = [doc.metadata.get('source', 'Unknown') for doc in source_docs]
            
            # Estimate tokens (rough approximation)
            tokens_used = len(answer.split()) + len(question.split())
            
            # Calculate confidence (based on source relevance)
            confidence = min(len(source_docs) / 3.0, 1.0)
            
            # Store metrics
            metrics = QueryMetrics(
                query=question,
                response=answer,
                response_time=response_time,
                tokens_used=tokens_used,
                sources=sources,
                timestamp=datetime.now().isoformat(),
                confidence_score=confidence
            )
            self.metrics_history.append(metrics)
            
            return {
                'answer': answer,
                'sources': sources,
                'response_time': response_time,
                'confidence': confidence,
                'source_documents': source_docs
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'response_time': time.time() - start_time,
                'confidence': 0.0,
                'source_documents': []
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.metrics_history:
            return {}
        
        avg_response_time = sum(m.response_time for m in self.metrics_history) / len(self.metrics_history)
        avg_confidence = sum(m.confidence_score for m in self.metrics_history) / len(self.metrics_history)
        total_queries = len(self.metrics_history)
        
        return {
            'total_queries': total_queries,
            'avg_response_time': avg_response_time,
            'avg_confidence': avg_confidence,
            'metrics_history': [asdict(m) for m in self.metrics_history]
        }


def create_vector_database(data_dir: str, output_dir: str):
    """Create FAISS vector database from PDF documents"""
    st.info("Creating vector database from documents...")
    
    # Load documents
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(output_dir)
    
    st.success(f"Vector database created with {len(texts)} chunks!")
    return True


def plot_metrics(metrics_data: List[Dict]):
    """Create visualizations for performance metrics"""
    if not metrics_data:
        return None
    
    # Response time trend
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=[m['response_time'] for m in metrics_data],
        mode='lines+markers',
        name='Response Time',
        line=dict(color='#1f77b4')
    ))
    fig1.update_layout(
        title="Response Time Trend",
        xaxis_title="Query Number",
        yaxis_title="Time (seconds)",
        height=300
    )
    
    # Confidence score distribution
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=[m['confidence_score'] for m in metrics_data],
        name='Confidence Score',
        marker_color='#2ca02c'
    ))
    fig2.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Query Number",
        yaxis_title="Confidence",
        height=300
    )
    
    return fig1, fig2


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Medical Chatbot - LLaMA 2",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .source-box {
            background-color: #e8f4f8;
            padding: 0.5rem;
            border-left: 3px solid #1f77b4;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Medical AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*Powered by LLaMA 2, RAG, and Vector Search*")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        model_path = st.text_input(
            "Model Path",
            value="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
            help="Path to LLaMA 2 model file"
        )
        
        db_path = st.text_input(
            "Vector DB Path",
            value="vectorstore/db_faiss",
            help="Path to FAISS vector database"
        )
        
        st.divider()
        
        # Database creation
        st.subheader("Database Management")
        with st.expander("Create New Vector DB"):
            data_dir = st.text_input("Data Directory", value="data/")
            output_dir = st.text_input("Output Directory", value="vectorstore/db_faiss")
            
            if st.button("Create Database"):
                create_vector_database(data_dir, output_dir)
        
        st.divider()
        
        # System info
        st.subheader("System Info")
        
        # Check device status
        import torch
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        device_status = "Apple GPU (MPS)" if has_mps else "CPU only"
        
        st.info(f"""
        **System:**
        - Device: {device_status}
        - Platform: macOS (ARM64)
        """)
        
        with st.expander("GPU/CPU Status"):
            st.write(f"""
            **Current Configuration:**
            - Embeddings: **Using Apple GPU (MPS)** ✓ - Fast
            - LLaMA Model: **CPU Only** ❌ - Slow
            
            **Why LLaMA Model Uses CPU:**
            - GGML format (CTransformers) is CPU-only by design
            - Does NOT support GPU acceleration (CUDA or Metal)
            - This is a format limitation, not a configuration issue
            
            **Performance:**
            - Document search: Fast (GPU-accelerated)
            - AI responses: Slow (CPU-only, 30-90 seconds)
            
            **Model:** Q8_0 (6.7GB) - Higher quality, slower
            """)
        
        with st.expander("Speed Up Your Chatbot"):
            st.write("""
            **Use Q4_0 Model (3.5GB) for 3-5x faster responses:**
            
            1. Download smaller model:
            ```
            huggingface-cli download TheBloke/Llama-2-7B-Chat-GGML \\
                llama-2-7b-chat.ggmlv3.q4_0.bin \\
                --local-dir models/
            ```
            
            2. Update Model Path in sidebar to:
            ```
            models/llama-2-7b-chat.ggmlv3.q4_0.bin
            ```
            
            3. Restart the app
            
            **Trade-off:** Slightly lower quality, but much faster!
            """)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.chat_history = []
        st.session_state.initialized = False
    
    # Initialize button
    if not st.session_state.initialized:
        st.info("Click the button below to initialize the chatbot. This may take 1-2 minutes.")
        
        if st.button("Initialize Chatbot", type="primary", use_container_width=True):
            progress_container = st.container()
            with progress_container:
                status_msg = st.empty()
                error_msg = st.empty()
            
            try:
                status_msg.info("Starting initialization... Please wait, this may take a few minutes.")
                
                chatbot = MedicalChatbot(model_path, db_path)
                
                # Initialize with progress updates
                if chatbot.initialize():
                    status_msg.success("Chatbot initialized successfully!")
                    st.session_state.chatbot = chatbot
                    st.session_state.initialized = True
                    st.rerun()
                else:
                    error_msg.error("Failed to initialize chatbot. Check error messages above for details.")
                    status_msg.empty()
                    
            except Exception as e:
                error_msg.error(f"Initialization failed with error: {str(e)}")
                status_msg.empty()
                import traceback
                with st.expander("Detailed Error"):
                    st.code(traceback.format_exc())
    
    # Main interface
    if st.session_state.initialized:
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Chat", "Analytics", "About"])
        
        with tab1:
            # Chat interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Ask Your Medical Question")
                
                # Check if chatbot is ready
                if st.session_state.chatbot is None or st.session_state.chatbot.llm is None:
                    st.error("⚠️ Chatbot not properly initialized. Please click 'Initialize Chatbot' again.")
                    if st.button("Re-initialize Chatbot"):
                        st.session_state.initialized = False
                        st.rerun()
                    st.stop()
                
                # Query input
                question = st.text_area(
                    "Your Question:",
                    height=100,
                    placeholder="E.g., What are the symptoms of diabetes?",
                    disabled=False
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    submit = st.button("Get Answer", type="primary", use_container_width=True)
                with col_btn2:
                    clear = st.button("Clear History", use_container_width=True)
                
                if clear:
                    st.session_state.chat_history = []
                    st.rerun()
                
                # Process query
                if submit and question:
                    # Show clear feedback about wait time
                    start_time = time.time()
                    status_placeholder = st.empty()
                    
                    with st.spinner("Processing your question..."):
                        status_placeholder.info("**Step 1/2: Searching medical documents...** This should take 1-2 seconds.")
                        
                        try:
                            status_placeholder.info("**Step 2/2: Generating AI response...** This may take 30-90 seconds on CPU. Please wait...")
                            
                            # Check memory before query
                            import psutil
                            memory = psutil.virtual_memory()
                            if memory.percent > 90:
                                st.warning(f"WARNING: System memory is {memory.percent:.1f}% full. This may cause slow performance. Close other apps if possible.")
                            
                            # This will take 30-90 seconds on CPU - it's a blocking call
                            # Note: If this takes more than 2-3 minutes, there's likely an issue
                            result = st.session_state.chatbot.query(question)
                            
                            # Store result
                            st.session_state.chat_history.append({
                                'question': question,
                                'result': result
                            })
                            
                            # Show completion message
                            elapsed_time = time.time() - start_time
                            response_time = result.get('response_time', elapsed_time)
                            
                            if response_time > 120:
                                st.error(f"WARNING: Query took {int(response_time)} seconds - this is unusually slow!")
                                st.warning("**Troubleshooting:**\n1. Your system may be low on memory\n2. Other apps may be using CPU\n3. Consider using the Q4_0 model for 3-5x speedup\n4. Close other applications and try again")
                            elif response_time > 90:
                                st.warning(f"Query took {int(response_time)} seconds. For faster responses, download the Q4_0 model (see sidebar instructions).")
                            
                            status_placeholder.empty()
                            
                        except TimeoutError as te:
                            status_placeholder.empty()
                            st.error(f"Timeout: {str(te)}")
                            result = {
                                'answer': f"Query timeout after 3 minutes.\n\n**This is taking too long. Try:**\n1. Restart the Streamlit app\n2. Close other applications to free up RAM/CPU\n3. Check Activity Monitor - is Python using lots of memory?\n4. Consider using the smaller Q4_0 model\n5. Try a simpler, shorter question",
                                'sources': [],
                                'response_time': time.time() - start_time,
                                'confidence': 0.0,
                                'source_documents': []
                            }
                            st.session_state.chat_history.append({
                                'question': question,
                                'result': result
                            })
                        except Exception as e:
                            status_placeholder.empty()
                            error_msg = str(e)
                            st.error(f"Error: {error_msg}")
                            
                            result = {
                                'answer': f"Error processing query: {error_msg}\n\n**Troubleshooting:**\n1. Check if you have at least 8GB free RAM\n2. Check Activity Monitor for memory usage\n3. Make sure the model file exists\n4. Try restarting the application\n5. Consider using Q4_0 model (smaller, faster)",
                                'sources': [],
                                'response_time': time.time() - start_time,
                                'confidence': 0.0,
                                'source_documents': []
                            }
                            
                            st.session_state.chat_history.append({
                                'question': question,
                                'result': result
                            })
                
                # Display chat history
                if st.session_state.chat_history:
                    st.divider()
                    st.subheader("Conversation History")
                    
                    for i, chat in enumerate(reversed(st.session_state.chat_history)):
                        with st.expander(f"Q: {chat['question'][:100]}...", expanded=(i==0)):
                            st.markdown("**Question:**")
                            st.write(chat['question'])
                            
                            st.markdown("**Answer:**")
                            st.write(chat['result']['answer'])
                            
                            # Metrics
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Response Time", f"{chat['result']['response_time']:.2f}s")
                            with col_m2:
                                st.metric("Confidence", f"{chat['result']['confidence']:.0%}")
                            with col_m3:
                                st.metric("Sources", len(chat['result']['sources']))
                            
                            # Sources
                            if chat['result']['sources']:
                                st.markdown("**Sources:**")
                                for src in chat['result']['sources']:
                                    st.markdown(f'<div class="source-box">{Path(src).name}</div>', 
                                              unsafe_allow_html=True)
            
            with col2:
                st.subheader("Quick Tips")
                st.info("""
                **How to ask questions:**
                - Be specific and clear
                - Include relevant symptoms
                - Ask one question at a time
                - Provide context when needed
                
                **Example questions:**
                - What are symptoms of flu?
                - How to treat a headache?
                - What causes high blood pressure?
                """)
                
                st.info("""
                **Performance Note:**
                Running on CPU mode (macOS):
                - Expected response time: 15-30 seconds
                - This is normal for CPU inference
                - For faster responses, consider:
                  - Using a smaller model (Q4_0 instead of Q8_0)
                  - Deploying on a Linux server with GPU
                """)
                
                st.warning("""
                **Disclaimer:**
                This is an AI assistant for informational purposes only. 
                Always consult healthcare professionals for medical advice.
                """)
        
        with tab2:
            st.subheader("Performance Analytics")
            
            metrics_summary = st.session_state.chatbot.get_metrics_summary()
            
            if metrics_summary:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Queries",
                        metrics_summary['total_queries'],
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Avg Response Time",
                        f"{metrics_summary['avg_response_time']:.2f}s",
                        delta=None
                    )
                with col3:
                    st.metric(
                        "Avg Confidence",
                        f"{metrics_summary['avg_confidence']:.0%}",
                        delta=None
                    )
                
                st.divider()
                
                # Charts
                if len(metrics_summary['metrics_history']) > 0:
                    fig1, fig2 = plot_metrics(metrics_summary['metrics_history'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("Detailed Metrics")
                st.dataframe(
                    [{
                        'Query': m['query'][:50] + '...',
                        'Response Time': f"{m['response_time']:.2f}s",
                        'Confidence': f"{m['confidence_score']:.0%}",
                        'Sources': len(m['sources']),
                        'Timestamp': m['timestamp'][:19]
                    } for m in metrics_summary['metrics_history']],
                    use_container_width=True
                )
            else:
                st.info("No metrics available yet. Start asking questions!")
        
        with tab3:
            st.subheader("About This Project")
            
            st.markdown("""
            ### Features
            
            **LLaMA 2 Integration**: Powered by quantized LLaMA 2 7B model for efficient CPU inference
            
            **Retrieval-Augmented Generation (RAG)**: Combines vector search with LLM generation
            
            **FAISS Vector Database**: Fast similarity search for relevant medical documents
            
            **Advanced Prompt Engineering**: Optimized prompts for medical context
            
            **Performance Metrics**: Track response times, confidence scores, and system performance
            
            **Source Attribution**: Every answer includes document sources
            
            **Scalable Architecture**: Modular design for easy integration
            
            ### Technology Stack
            
            - **LLM**: LLaMA 2 7B (GGML quantized)
            - **Framework**: LangChain
            - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
            - **Vector DB**: FAISS
            - **UI**: Streamlit
            - **Visualization**: Plotly
            
            ### Project Structure
            ```
            medical-chatbot/
            ├── data/              # Medical PDF documents
            ├── models/            # LLaMA 2 model files
            ├── vectorstore/       # FAISS vector database
            ├── app.py            # Main Streamlit application
            ├── ingest.py         # Document ingestion script
            └── requirements.txt  # Dependencies
            ```
            
            ### Setup Instructions
            
            1. Install dependencies: `pip install -r requirements.txt`
            2. Download LLaMA 2 model from HuggingFace
            3. Place medical PDFs in `data/` directory
            4. Create vector database using sidebar tool
            5. Initialize chatbot and start asking questions!
            
            ### Performance Optimization
            
            - **Quantization**: Using GGML Q8 quantization for 4x smaller model size
            - **Chunk Size**: Optimized to 500 characters with 50 character overlap
            - **Retrieval**: Top-K=3 for balance between context and speed
            - **Temperature**: Set to 0.2 for more deterministic medical responses
            
            ### Future Enhancements
            
            - Multi-modal support (images, lab reports)
            - Fine-tuning on medical datasets
            - Integration with medical APIs
            - Multi-language support
            - Voice input/output
            """)
            
            st.divider()
            
            st.markdown("""
            ### References & Resources
            
            - [LangChain Documentation](https://python.langchain.com/)
            - [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
            - [FAISS Documentation](https://faiss.ai/)
            - [Sentence Transformers](https://www.sbert.net/)
            
            ---
            
            **Built for Healthcare AI**
            """)
    
    else:
        # Welcome screen
        st.info("Configure settings in the sidebar and click 'Initialize Chatbot' to start!")


if __name__ == "__main__":
    main()

