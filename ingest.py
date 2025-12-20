"""
Document Ingestion Script for Medical Chatbot
Creates FAISS vector database from PDF documents
"""

import os
from pathlib import Path
from typing import List
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Try to import from langchain_community for compatibility
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    # Fallback to old langchain imports
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS


class DocumentIngestor:
    """Handle document loading and vector database creation"""
    
    def __init__(
        self, 
        data_path: str = "data",
        db_path: str = "vectorstore/db_faiss",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.data_path = data_path
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = None
        self.documents = []
        self.text_chunks = []
        
    def load_documents(self) -> List:
        """Load PDF documents from data directory"""
        print(f"Loading documents from {self.data_path}...")
        
        # Check if directory exists
        if not Path(self.data_path).exists():
            raise ValueError(f"Data directory not found: {self.data_path}")
        
        # Load PDFs
        loader = DirectoryLoader(
            self.data_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} documents")
        
        return self.documents
    
    def split_documents(self) -> List:
        """Split documents into chunks"""
        print(f"Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Created {len(self.text_chunks)} text chunks")
        
        return self.text_chunks
    
    def create_embeddings(self):
        """Initialize embedding model"""
        print("Initializing embedding model...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("Embedding model initialized")
        
    def create_vector_store(self):
        """Create and save FAISS vector database"""
        print("Creating FAISS vector database...")
        
        # Create vector store
        vectorstore = FAISS.from_documents(
            documents=self.text_chunks,
            embedding=self.embeddings
        )
        
        # Create output directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        vectorstore.save_local(self.db_path)
        print(f"Vector database saved to {self.db_path}")
        
        # Print statistics
        self.print_statistics()
        
    def print_statistics(self):
        """Print database statistics"""
        print("\n" + "="*50)
        print("DATABASE STATISTICS")
        print("="*50)
        print(f"Total Documents: {len(self.documents)}")
        print(f"Total Chunks: {len(self.text_chunks)}")
        print(f"Chunk Size: {self.chunk_size} characters")
        print(f"Chunk Overlap: {self.chunk_overlap} characters")
        print(f"Embedding Model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"Database Location: {self.db_path}")
        print("="*50 + "\n")
        
    def run(self):
        """Execute full ingestion pipeline"""
        print("\nStarting document ingestion pipeline...\n")
        
        try:
            # Step 1: Load documents
            self.load_documents()
            
            if not self.documents:
                print("No documents found!")
                return False
            
            # Step 2: Split documents
            self.split_documents()
            
            # Step 3: Create embeddings
            self.create_embeddings()
            
            # Step 4: Create vector store
            self.create_vector_store()
            
            print("Ingestion pipeline completed successfully!\n")
            return True
            
        except Exception as e:
            print(f"Error during ingestion: {str(e)}")
            return False


def main():
    """Main execution function"""
    
    # Configuration
    DATA_PATH = "data"
    DB_PATH = "vectorstore/db_faiss"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   Medical Chatbot - Document Ingestion Pipeline      ║
    ║   Creates FAISS Vector Database from PDF Documents   ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Create ingestor
    ingestor = DocumentIngestor(
        data_path=DATA_PATH,
        db_path=DB_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Run pipeline
    success = ingestor.run()
    
    if success:
        print("Ready to use! Run the Streamlit app with: streamlit run app.py")
    else:
        print("Pipeline failed. Please check errors above.")


if __name__ == "__main__":
    main()

