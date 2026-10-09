"""
utils.py - Utility functions for logging, file hashing, validation, and session state management.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import streamlit as st
from config import ALLOWED_EXTENSIONS, HASHES_FILE, DATA_DIR

# Setup application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LocalRAGChatbot")


def setup_environment() -> None:
    """Ensure data directories and storage files exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HASHES_FILE.exists():
        save_processed_hashes({})
    logger.info("Environment initialized successfully.")


def get_file_hash(file_bytes: bytes) -> str:
    """
    Calculate SHA256 hash of file content for duplicate detection.
    
    Args:
        file_bytes: Byte content of the file.
        
    Returns:
        HEX string of SHA256 hash.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def load_processed_hashes() -> Dict[str, Dict[str, str]]:
    """
    Load stored document hashes and metadata from disk.
    
    Returns:
        Dictionary mapping file hash to metadata (filename, hash, timestamp, etc.).
    """
    if not HASHES_FILE.exists():
        return {}
    try:
        with open(HASHES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading processed hashes: {e}")
        return {}


def save_processed_hashes(hashes_data: Dict[str, Dict[str, str]]) -> None:
    """
    Save document hashes to JSON file.
    
    Args:
        hashes_data: Dictionary mapping file hashes to metadata.
    """
    try:
        with open(HASHES_FILE, "w", encoding="utf-8") as f:
            json.dump(hashes_data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving processed hashes: {e}")


def is_valid_file(filename: str) -> bool:
    """
    Check if the file extension is supported.
    
    Args:
        filename: Name of the uploaded file.
        
    Returns:
        True if valid extension, False otherwise.
    """
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def format_error_message(error: Exception, provider: str = "") -> str:
    """
    Format exception into a user-friendly error string without exposing full stack traces.
    
    Args:
        error: Caught exception.
        provider: Active LLM provider name if relevant.
        
    Returns:
        Human-readable error description.
    """
    error_str = str(error)
    if "api_key" in error_str.lower() or "authentication" in error_str.lower():
        return f"⚠️ Authentication failed for {provider}. Please check your API key in settings or `.env` file."
    elif "connection" in error_str.lower() or "connect" in error_str.lower():
        if provider == "Ollama":
            return "⚠️ Cannot connect to Ollama. Please ensure Ollama service is running (`ollama serve`)."
        return f"⚠️ Network connection error while communicating with {provider}."
    elif "not found" in error_str.lower() and provider == "Ollama":
        return f"⚠️ The requested Ollama model was not found. Download it using `ollama pull <model_name>`."
    elif "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
        return f"⚠️ API rate limit reached for {provider}. Please wait a few moments before asking again."
    else:
        return f"⚠️ An error occurred: {error_str}"


def initialize_session_state() -> None:
    """Initialize Streamlit session state keys if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0
    if "status" not in st.session_state:
        st.session_state.status = "🟢 Ready"
    if "status_message" not in st.session_state:
        st.session_state.status_message = "System ready for queries."
    if "last_retrieved_sources" not in st.session_state:
        st.session_state.last_retrieved_sources = []
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False
