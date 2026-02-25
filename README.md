# 🚀 SimpleRAGPDF-Demo

A simple yet powerful **Retrieval-Augmented Generation (RAG)** system for PDF documents.

Extract text from PDFs → chunk it intelligently → generate embeddings → answer questions using a local LLM. **All without LangChain!**

---

## ⭐ Key Features

- ✨ **No LangChain Required** - Custom recursive text splitter with same behavior
- 🚀 **Lightweight** - 81M parameter model (distilgpt2), runs on CPU
- 📄 **PyMuPDF** - Fast and reliable PDF text extraction  
- 🧠 **Semantic Search** - SentenceTransformers embeddings
- 🔍 **FAISS** - Lightning-fast vector similarity (<50ms per query)
- ⏱️ **Performance Metrics** - Built-in timing for all operations
- 💻 **Production Code** - Modular, documented, extensible
- 📊 **Jupyter & CLI** - Multiple ways to use

---

## 🎯 Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run interactive demo
python main.py

# 3. Provide PDF path and ask questions!
```

**For Jupyter:**
```bash
jupyter notebook SimpleRAGPDF_Notebook.ipynb
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[DEMO_GUIDE.md](DEMO_GUIDE.md)** | 🎤 How to present this project |
| **[EXAMPLES.md](EXAMPLES.md)** | 💡 Code examples & usage patterns |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | 🏗️ Deep dive into design |
| **[ORGANIZATION.md](ORGANIZATION.md)** | 📦 Code structure overview |

---

## 👀 See It In Action

### First, understand what it does:
```bash
python demo.py
```
Shows system overview with ASCII diagrams and performance expectations.

### Then, try it interactively:
```bash
python main.py
```

**Output Example:**
```
Extracting text from document.pdf...
Extracted 45823 characters
Splitting text into chunks...
Created 52 chunks
Building embedding index...
Index built in 3.42s

> Ask a question: What is this about?

Searching and generating answer...
Tokenize: 0.156s | Generate: 2.341s | Input tokens: 287

Answer:
This document discusses advanced machine learning techniques...
```

---

## 📖 How It Works

```
1. PDF EXTRACTION (PyMuPDF)
   Input: PDF file → Output: Raw text

2. TEXT CHUNKING (Custom, intelligent splitting)
   900 chars/chunk with 150 char overlap

3. EMBEDDINGS (SentenceTransformers)
   Semantic understanding of chunks

4. FAISS INDEX (Fast vector search)
   <50ms per query

5. RETRIEVAL (Top-K similarity search)
   Find 3 most relevant chunks by default

6. GENERATION (distilgpt2 LLM)
   Generate answer using context + question
```

**See detailed architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🏗️ Project Structure

```
SimpleRAGPDF-Demo/
├── 🚀 main.py                          Interactive CLI entry point
├── 🎬 demo.py                          System overview & guide
├── 📓 SimpleRAGPDF_Notebook.ipynb       Jupyter notebook (Colab compatible)
├── 📋 requirements.txt                  Dependencies
├── 📖 README.md                         This file
├── 📚 DEMO_GUIDE.md                     How to present this
├── 💡 EXAMPLES.md                       Code examples
├── 🏗️  ARCHITECTURE.md                  Deep technical dive
├── 📦 ORGANIZATION.md                   Code structure
└── src/
    ├── config.py                     ⚙️  Configuration
    ├── pdf_processor.py              📄 PDF extraction & chunking
    ├── embeddings.py                 🧠 Embeddings & FAISS
    ├── llm.py                        🤖 Model inference
    └── rag.py                        🔗 Main orchestrator
```

---

## 💻 Code Example

### Basic Usage
```python
from src.rag import SimpleRAG

# Initialize
rag = SimpleRAG()

# Load PDF
rag.load_pdf("document.pdf")

# Ask questions
result = rag.ask("What is the main topic?")
print(result['answer'])
```

### Advanced Usage
```python
from src.rag import SimpleRAG

rag = SimpleRAG(device="cuda")  # Use GPU
rag.load_pdf("document.pdf", verbose=True)

result = rag.ask(
    question="Summarize key findings",
    top_k=5,              # Retrieve 5 chunks
    max_new_tokens=200,   # Longer answer
    show_context=True     # Show retrieved text
)

print(f"Answer: {result['answer']}")
print(f"Timing: {result['timing']}")
print(f"Sources: {result['context']}")
```

**More examples:** [EXAMPLES.md](EXAMPLES.md)

---

## 🎯 For Demo/Presentation

Perfect for showing RAG principles in action!

**Recommended flow:**
1. Run `python demo.py` (2 min) - Overview
2. Show code structure (2 min)
3. Run `python main.py` (15 min) - Live demo
4. Q&A (5 min)

**Total: ~20 minutes**

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed presentation tips, talking points, and Q&A prep.

---

## 📊 Performance

| Metric | CPU | GPU |
|--------|-----|-----|
| **PDF extraction** (10-page) | ~0.5s | ~0.5s |
| **Text chunking** (50 chunks) | ~0.1s | ~0.1s |
| **Embeddings** (50 chunks) | ~2-5s | ~0.2-0.5s |
| **Query retrieval** | <50ms | <50ms |
| **Text generation** (120 tokens) | 1-3s | 0.2-0.5s |
| **Total per query** | ~1.5-3s | ~0.4-0.7s |

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Text splitting
CHUNK_SIZE = 900              # Characters per chunk
CHUNK_OVERLAP = 150           # Overlap between chunks
MIN_CHUNK_CHARS = 200         # Minimum chunk size

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Fast & semantic
LLM_MODEL = "distilgpt2"                # Lightweight generation

# Generation
MAX_INPUT_TOKENS = 512        # Input prompt size
MAX_NEW_TOKENS = 120          # Max output length
TOP_K = 3                     # Chunks to retrieve
```

---

## 📦 Dependencies

- `pymupdf` - PDF text extraction
- `sentence-transformers` - Semantic embeddings
- `faiss-cpu` - Vector similarity search
- `transformers` - HuggingFace models
- `torch` - Deep learning framework
- `numpy` - Numerical operations

All specified in `requirements.txt`

---

## 💡 Tips & Tricks

- For **GPU acceleration**: `faiss-gpu` instead of `faiss-cpu`
- For **better quality**: Try different embedding models (trade-off with speed)
- For **larger answers**: Increase `max_new_tokens`
- For **more context**: Increase `top_k` retrieval
- For **faster results**: Reduce `chunk_size` and `top_k`

---

## ⚠️ Limitations

- Works with English documents (transformer models are English-focused)
- Must fit document in system memory
- Requires internet for first-time model downloads
- Answers depend on document quality and question clarity

---

## 🔄 Model Alternatives

Easily swap models in `src/config.py`:

```python
# Larger embedding (better quality, slower)
"all-mpnet-base-v2"           # 109M params

# Faster embedding
"all-MiniLM-L6-v2"            # 22M params (default)

# Larger LLM (better answers, slower, needs GPU)
"gpt2"                        # 124M params
"distilgpt2"                  # 82M params (default)
```

---

## 🛣️ Future Enhancements

- [ ] Multi-format support (Word, PowerPoint, HTML)
- [ ] Multi-language support
- [ ] Persistent vector database integration
- [ ] Web interface
- [ ] Streaming response generation
- [ ] Document Q&A with citations
- [ ] Batch processing

---

## 🤝 How to Extend

**Swap components:**
```python
# Use different embedding model
from src.embeddings import EmbeddingIndex
index = EmbeddingIndex("all-mpnet-base-v2")

# Use different LLM
from src.llm import LLMGenerator
llm = LLMGenerator("gpt2", device="cuda")
```

**Modify pipeline:**
```python
# More context chunks
result = rag.ask(query, top_k=10)

# Longer answers
result = rag.ask(query, max_new_tokens=300)
```

**See:** [ARCHITECTURE.md](ARCHITECTURE.md#extensibility-points)

---

## 🧠 Learning Resources

This project is great for learning:
- ✅ RAG (Retrieval-Augmented Generation) principles
- ✅ FAISS vector databases
- ✅ Sentence Transformers embeddings
- ✅ Text chunking strategies
- ✅ LLM inference optimization
- ✅ Production Python code structure

---

## ❓ FAQ

**Q: Why not use LangChain?**  
A: This teaches the core concepts clearly without abstraction layers. LangChain is great for production but adds complexity for learning.

**Q: Can I use my custom LLM?**  
A: Yes! Modify `src/llm.py` and `src/config.py` to load any HuggingFace model.

**Q: How to use with GPU?**  
A: Pass `device="cuda"` to SimpleRAG: `rag = SimpleRAG(device="cuda")`

**Q: Works with non-English PDFs?**  
A: Current models are English-trained. Future enhancement: multilingual models.

**Q: How much memory needed?**  
A: ~700MB for models + ~100MB per 50 chunks of document.

---

## 📝 License

MIT

---

## 🎬 Ready to Demo?

1. `pip install -r requirements.txt`
2. `python demo.py` (understand the system)
3. `python main.py` (interactive demo)
4. Read [DEMO_GUIDE.md](DEMO_GUIDE.md) for presentation tips

**Show this to others!** It's educational, practical, and impressive. 🚀

---

**Questions?** See [EXAMPLES.md](EXAMPLES.md) for more usage patterns or [ARCHITECTURE.md](ARCHITECTURE.md) for technical details.
