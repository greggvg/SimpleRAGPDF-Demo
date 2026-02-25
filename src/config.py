"""Configuration constants for SimpleRAGPDF"""

# Text Splitting Configuration
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 200

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Configuration
LLM_MODEL = "distilgpt2"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 120

# Retrieval Configuration
TOP_K = 3
