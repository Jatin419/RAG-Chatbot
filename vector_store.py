"""
vector_store.py - Vector database management using HuggingFace embeddings and ChromaDB.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import streamlit as st
from langchain_core.documents import Document

# Compatible imports for HuggingFace embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Compatible imports for Chroma
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from config import EMBEDDING_MODEL_NAME, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME, HASHES_FILE
from utils import logger, save_processed_hashes


@st.cache_resource(show_spinner="Loading HuggingFace Embedding Model...")
def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load HuggingFace local embedding model (sentence-transformers/all-MiniLM-L6-v2).
    Cached via Streamlit resource caching to prevent reloading across re-runs.
    
    Returns:
        HuggingFaceEmbeddings object.
    """
    logger.info(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def load_vector_store(embedding_model: HuggingFaceEmbeddings) -> Optional[Chroma]:
    """
    Load existing persistent Chroma vector database if directory exists.
    
    Args:
        embedding_model: Loaded HuggingFace embedding model instance.
        
    Returns:
        Chroma vector store instance or None if not initialized.
    """
    db_path = str(CHROMA_DB_DIR)
    if not CHROMA_DB_DIR.exists() or not os.listdir(db_path):
        logger.info("Chroma DB directory does not exist or is empty.")
        return None
        
    try:
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=db_path
        )
        logger.info("Successfully loaded existing ChromaDB vector store.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load Chroma vector store: {e}")
        return None


def add_documents(
    vector_store: Optional[Chroma],
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings
) -> Chroma:
    """
    Add document chunks to vector store. Initializes new ChromaDB if vector_store is None.
    
    Args:
        vector_store: Active Chroma vector store instance or None.
        chunks: List of Document chunks to add.
        embedding_model: Loaded HuggingFace embedding model.
        
    Returns:
        Updated Chroma vector store instance.
    """
    db_path = str(CHROMA_DB_DIR)
    
    if not chunks:
        logger.warning("No chunks provided to add to vector store.")
        if vector_store is None:
            vector_store = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_model,
                persist_directory=db_path
            )
        return vector_store

    if vector_store is None:
        logger.info(f"Creating new Chroma vector database with {len(chunks)} chunks.")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=db_path
        )
    else:
        logger.info(f"Adding {len(chunks)} chunks to existing Chroma vector store.")
        vector_store.add_documents(chunks)
        
    return vector_store


def retrieve_documents(
    vector_store: Optional[Chroma],
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Retrieve top-k most relevant document chunks for a question.
    
    Args:
        vector_store: Chroma vector store instance.
        query: User query question string.
        k: Number of relevant chunks to retrieve.
        
    Returns:
        List of retrieved Document objects.
    """
    if vector_store is None:
        logger.warning("Vector store is not initialized. Cannot retrieve documents.")
        return []

    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        results = retriever.invoke(query)
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:30]}...'")
        return results
    except Exception as e:
        logger.error(f"Error during vector retrieval: {e}")
        return []


def reset_vector_store() -> bool:
    """
    Wipe out persistent ChromaDB files and reset duplicate document hashes.
    
    Returns:
        True if successfully reset, False otherwise.
    """
    try:
        db_path = str(CHROMA_DB_DIR)
        if CHROMA_DB_DIR.exists():
            shutil.rmtree(db_path)
            CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("ChromaDB directory deleted and recreated.")
            
        save_processed_hashes({})
        logger.info("Processed file hashes reset.")
        return True
    except Exception as e:
        logger.error(f"Failed to reset vector store: {e}")
        return False
