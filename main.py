"""
Main entry point for SimpleRAGPDF interactive demo.

This script provides an interactive interface for asking questions about a PDF document
using retrieval-augmented generation with FAISS embeddings and a small LLM.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag import SimpleRAG


def main():
    """Main interactive loop"""
    
    print("=" * 60)
    print("SimpleRAGPDF - Interactive Q&A Demo")
    print("=" * 60)
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag = SimpleRAG()
    
    # Load PDF
    pdf_path = input("\nEnter path to PDF file: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    num_chunks = rag.load_pdf(pdf_path, verbose=True)
    print(f"\nSuccessfully loaded PDF with {num_chunks} chunks.")
    
    # Test with initial question
    print("\n" + "=" * 60)
    print("Testing with sample question...")
    print("=" * 60)
    
    result = rag.ask("What is this document about?", top_k=3, verbose=True)
    print(f"\nAnswer: {result['answer']}\n")
    
    # Interactive loop
    print("=" * 60)
    print("Interactive Mode - Ask questions about the PDF")
    print("Type 'exit' to quit, 'context' to show retrieved context")
    print("=" * 60)
    
    show_context = False
    
    while True:
        query = input("\n> Ask a question: ").strip()
        
        if not query:
            continue
        
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        if query.lower() == "context":
            show_context = not show_context
            print(f"Context display: {'ON' if show_context else 'OFF'}")
            continue
        
        print("\nSearching and generating answer...\n")
        
        try:
            result = rag.ask(
                query,
                top_k=3,
                show_context=show_context,
                verbose=True
            )
            print(f"Answer:\n{result['answer']}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
