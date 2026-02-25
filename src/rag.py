"""Main RAG (Retrieval-Augmented Generation) pipeline"""

from typing import Optional
from .pdf_processor import extract_text_from_pdf, recursive_text_splitter
from .embeddings import EmbeddingIndex
from .llm import LLMGenerator
from . import config


class SimpleRAG:
    """Simple RAG system combining PDF extraction, embeddings, retrieval, and LLM"""
    
    def __init__(
        self,
        embedding_model: str = config.EMBEDDING_MODEL,
        llm_model: str = config.LLM_MODEL,
        device: Optional[str] = None
    ):
        """
        Initialize RAG system.
        
        Args:
            embedding_model: Name of embedding model
            llm_model: Name of LLM model
            device: Device to use ('cuda' or 'cpu')
        """
        self.embedding_index = EmbeddingIndex(embedding_model)
        self.llm = LLMGenerator(llm_model, device=device)
        self.chunks = None
        
    def load_pdf(self, pdf_path: str, verbose: bool = True) -> int:
        """
        Load and process a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            verbose: Whether to print progress
            
        Returns:
            Number of chunks created
        """
        # Extract text from PDF
        if verbose:
            print(f"Extracting text from {pdf_path}...")
        raw_text = extract_text_from_pdf(pdf_path).strip()
        
        if verbose:
            print(f"Extracted {len(raw_text)} characters")
        
        # Split into chunks
        if verbose:
            print("Splitting text into chunks...")
        self.chunks = recursive_text_splitter(
            raw_text,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            min_chunk_chars=config.MIN_CHUNK_CHARS
        )
        
        if verbose:
            print(f"Created {len(self.chunks)} chunks")
        
        # Build embedding index
        if verbose:
            print("Building embedding index...")
        embed_time = self.embedding_index.build_index(self.chunks, show_progress=verbose)
        
        if verbose:
            print(f"Index built in {embed_time:.2f}s")
            stats = self.embedding_index.get_stats()
            print(f"Index stats: {stats}")
        
        return len(self.chunks)
    
    def ask(
        self,
        question: str,
        top_k: int = config.TOP_K,
        max_input_tokens: int = config.MAX_INPUT_TOKENS,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        show_context: bool = False,
        verbose: bool = True
    ) -> dict:
        """
        Ask a question about the loaded PDF.
        
        Args:
            question: Question to ask
            top_k: Number of chunks to retrieve
            max_input_tokens: Maximum input tokens for LLM
            max_new_tokens: Maximum new tokens to generate
            show_context: Whether to print retrieved context
            verbose: Whether to print timing info
            
        Returns:
            Dictionary with 'answer', 'context', and 'timing' keys
        """
        if self.chunks is None:
            raise ValueError("No PDF loaded. Call load_pdf() first.")
        
        # Retrieve relevant chunks
        context_chunks, distances = self.embedding_index.retrieve(question, top_k=top_k)
        context = "\n\n".join(context_chunks)
        
        if show_context:
            print(f"\nRetrieved context (distances: {distances}):\n{context[:500]}...\n")
        
        # Build prompt
        prompt = (
            "Use the context to answer the question.\n"
            "If the answer is not in the context, say: I do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
        
        # Generate answer
        answer, timing = self.llm.generate(
            prompt,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens
        )
        
        if verbose:
            print(f"Tokenize: {timing['tokenize']}s | Generate: {timing['generate']}s | Input tokens: {timing['input_tokens']}")
        
        return {
            "answer": answer,
            "context": context_chunks,
            "timing": timing
        }
