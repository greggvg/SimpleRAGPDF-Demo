## SimpleRAGPDF Project Organization - Quick Reference

### Directory Structure
```
SimpleRAGPDF-Demo/
├── main.py                      # ✨ Run this: python main.py
├── requirements.txt             # Install with: pip install -r requirements.txt
├── SimpleRAGPDF_Notebook.ipynb  # For Jupyter/Colab
├── .gitignore                   # Git ignore configuration
├── README.md                    # Full documentation
└── src/
    ├── __init__.py
    ├── config.py                # Configuration constants
    ├── pdf_processor.py         # PDF extraction & text chunking
    ├── embeddings.py            # Embedding & FAISS index
    ├── llm.py                   # Language model inference
    └── rag.py                   # Main RAG orchestrator
```

### Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the CLI:**
   ```bash
   python main.py
   ```
   - Enter PDF path when prompted
   - Ask questions interactively
   - Type 'exit' to quit

3. **Or use Jupyter:**
   ```bash
   jupyter notebook SimpleRAGPDF_Notebook.ipynb
   ```

### Key Modules

- **config.py**: Centralized settings (chunk size, models, etc.)
- **pdf_processor.py**: PyMuPDF text extraction + custom text splitter
- **embeddings.py**: EmbeddingIndex class for FAISS operations
- **llm.py**: LLMGenerator class for model inference
- **rag.py**: SimpleRAG class - main interface combining all components

### Code Organization Benefits

✅ **Modular**: Each component in separate file
✅ **Reusable**: Import SimpleRAG directly: `from src.rag import SimpleRAG`
✅ **Configurable**: All settings in config.py
✅ **Documented**: Full docstrings in every module
✅ **Tested**: Can be imported without LangChain dependency
✅ **Production-ready**: Proper error handling and logging

### Example Usage

```python
from src.rag import SimpleRAG

rag = SimpleRAG()
rag.load_pdf("document.pdf")
result = rag.ask("What is this about?")
print(result['answer'])
```
