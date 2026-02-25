"""Embedding model and FAISS index utilities"""

import time
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class EmbeddingIndex:
    """Manages embeddings and FAISS vector index"""
    
    def __init__(self, model_name: str):
        """
        Initialize embedding model and FAISS index.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None
        self.embedding_dim = None
        
    def build_index(self, chunks: List[str], show_progress: bool = True) -> float:
        """
        Build FAISS index from text chunks.
        
        Args:
            chunks: List of text chunks
            show_progress: Whether to show progress bar
            
        Returns:
            Time taken to compute embeddings in seconds
        """
        self.chunks = chunks
        
        t0 = time.time()
        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        t1 = time.time()
        
        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        
        return t1 - t0
        
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[str], np.ndarray]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (list of chunk texts, distances array)
        """
        if self.index is None or self.chunks is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        q_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)
        selected = [self.chunks[i] for i in indices[0]]
        
        return selected, distances[0]
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        if self.index is None:
            return {"built": False}
        
        return {
            "built": True,
            "num_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(self.chunks) if self.chunks else 0
        }
