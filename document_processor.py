"""
document_processor.py - Document loading, metadata assignment, text splitting, and duplicate detection.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import logger, get_file_hash, load_processed_hashes, save_processed_hashes, is_valid_file


def load_document(file_path: str, original_filename: str) -> List[Document]:
    """
    Load document content based on file extension and assign clean metadata.
    
    Args:
        file_path: Local file path on disk.
        original_filename: Original filename uploaded by the user.
        
    Returns:
        List of LangChain Document objects.
    """
    ext = Path(original_filename).suffix.lower()
    docs: List[Document] = []
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = original_filename
                doc.metadata["file_type"] = "pdf"
                # Keep page metadata intact; PyPDFLoader sets zero-indexed 'page'
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 0
                    
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = original_filename
                doc.metadata["file_type"] = "txt"
                doc.metadata["page"] = 0
                
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = original_filename
                doc.metadata["file_type"] = "docx"
                doc.metadata["page"] = 0

    except Exception as e:
        logger.error(f"Failed to load document {original_filename}: {e}")
        raise ValueError(f"Error loading {original_filename}: {str(e)}")
        
    # Filter out empty or whitespace-only documents
    valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    if not valid_docs:
        logger.warning(f"Document {original_filename} contained no readable text.")
        
    return valid_docs


def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Split loaded documents into smaller overlapping chunks for embedding.
    
    Args:
        documents: List of loaded Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.
        
    Returns:
        List of chunked Document objects preserving original metadata.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document pages into {len(chunks)} text chunks.")
    return chunks


def process_uploaded_files(
    uploaded_files: List[Any],
    chunk_size: int,
    chunk_overlap: int
) -> Tuple[List[Document], List[str], List[str]]:
    """
    Process list of Streamlit UploadedFile objects with duplicate checking.
    
    Args:
        uploaded_files: Streamlit uploaded file objects.
        chunk_size: Text splitter chunk size.
        chunk_overlap: Text splitter chunk overlap.
        
    Returns:
        Tuple of (all_chunks, newly_processed_filenames, skipped_filenames).
    """
    processed_hashes = load_processed_hashes()
    all_chunks: List[Document] = []
    newly_processed: List[str] = []
    skipped_files: List[str] = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        if not is_valid_file(filename):
            logger.warning(f"File {filename} has an unsupported extension.")
            continue
            
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset pointer
        
        file_hash = get_file_hash(file_bytes)
        
        # Duplicate check
        if file_hash in processed_hashes:
            logger.info(f"Duplicate document detected: {filename} (hash match)")
            skipped_files.append(filename)
            continue
            
        # Create temporary file to load
        ext = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            docs = load_document(temp_path, filename)
            if not docs:
                logger.warning(f"No text extracted from {filename}")
                continue
                
            chunks = split_documents(docs, chunk_size, chunk_overlap)
            if chunks:
                all_chunks.extend(chunks)
                newly_processed.append(filename)
                
                # Update processed hashes
                processed_hashes[file_hash] = {
                    "filename": filename,
                    "chunk_count": len(chunks),
                    "hash": file_hash
                }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Save updated hash store
    if newly_processed:
        save_processed_hashes(processed_hashes)

    return all_chunks, newly_processed, skipped_files
