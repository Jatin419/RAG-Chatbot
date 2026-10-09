"""
config.py - Application configuration and default parameters.
Contains constants for default settings, paths, models, and provider options.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
HASHES_FILE = DATA_DIR / "processed_hashes.json"

# Ensure essential directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# Default RAG settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.2

# Chunking constraints
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4000
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 1000

# Top-K constraints
MIN_TOP_K = 1
MAX_TOP_K = 10

# Embedding model settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "rag_documents"

# LLM Providers & Model Choices
LLM_PROVIDERS = {
    "Ollama": {
        "models": ["llama3.2", "llama3.1", "mistral", "gemma2", "qwen2.5"],
        "default_model": "llama3.2",
        "requires_api_key": False
    },
    "OpenAI": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "requires_api_key": True,
        "env_var": "OPENAI_API_KEY"
    },
    "Groq": {
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "default_model": "llama-3.3-70b-versatile",
        "requires_api_key": True,
        "env_var": "GROQ_API_KEY"
    }
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Memory settings
MAX_HISTORY_MESSAGES = 10
