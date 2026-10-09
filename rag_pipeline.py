"""
rag_pipeline.py - Core RAG flow: retrieval, context construction, prompt engineering, and LLM response generation.
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

from vector_store import retrieve_documents
from utils import logger, format_error_message


# Exact prompt required by specification
RAG_SYSTEM_PROMPT = """You are a document question-answering assistant.

Your job is to answer questions using only the provided retrieved context.

Rules:

1. Do not invent information.
2. Do not use unsupported facts.
3. If the answer is not present in the context, say:
   'I couldn't find relevant information in the uploaded documents.'
4. Use conversation history only to understand follow-up questions.
5. Give clear and concise answers.
6. Explain concepts simply when requested.
7. Do not claim that a source says something unless it appears in the retrieved context.

Retrieved Context:

{context}

Conversation History:

{history}

User Question:

{question}"""

NO_CONTEXT_RESPONSE = (
    "I couldn't find relevant information in the uploaded documents.\n\n"
    "Please try rephrasing your question or upload a document containing the required information."
)


def retrieve_context(vector_store: Any, question: str, k: int = 4) -> List[Document]:
    """
    Retrieve top-k relevant document chunks from Chroma vector store.
    
    Args:
        vector_store: Chroma vector store instance.
        question: User search question string.
        k: Number of chunks to retrieve.
        
    Returns:
        List of matching Document objects.
    """
    if vector_store is None:
        return []
    return retrieve_documents(vector_store, question, k)


def build_context(retrieved_docs: List[Document]) -> str:
    """
    Format retrieved documents into a clean context string for the LLM prompt.
    
    Args:
        retrieved_docs: List of retrieved Document chunks.
        
    Returns:
        Combined string of context blocks with source headers.
    """
    if not retrieved_docs:
        return "No relevant context found."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 0)
        file_type = doc.metadata.get("file_type", "")
        
        # Display page number 1-indexed for PDFs
        if file_type == "pdf" or "page" in doc.metadata:
            page_info = f" (Page {page + 1})"
        else:
            page_info = ""

        part = f"[Chunk {i} | Source: {source}{page_info}]\n{doc.page_content.strip()}"
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


def format_chat_history(chat_history: List[Dict[str, str]], max_messages: int = 6) -> str:
    """
    Format recent chat messages into a readable conversation history block.
    
    Args:
        chat_history: List of dicts containing 'role' and 'content'.
        max_messages: Maximum recent messages to include.
        
    Returns:
        Formatted history string.
    """
    if not chat_history:
        return "No previous conversation history."

    recent_messages = chat_history[-max_messages:]
    history_lines = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("\n", " ")
        history_lines.append(f"{role}: {content}")

    return "\n".join(history_lines)


def build_prompt(context: str, history: str, question: str) -> str:
    """
    Build the final prompt string using the system template.
    
    Args:
        context: Formatted retrieved documents context.
        history: Formatted conversation history.
        question: Current user question.
        
    Returns:
        Complete formatted prompt.
    """
    return RAG_SYSTEM_PROMPT.format(
        context=context,
        history=history,
        question=question
    )


def extract_deduplicated_sources(retrieved_docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract and deduplicate source citations from retrieved document chunks.
    
    Args:
        retrieved_docs: List of retrieved Document objects.
        
    Returns:
        List of unique source dicts containing 'source', 'page', 'snippet', and 'file_type'.
    """
    seen_keys = set()
    deduped_sources = []

    for doc in retrieved_docs:
        source_name = doc.metadata.get("source", "Unknown Document")
        raw_page = doc.metadata.get("page", 0)
        file_type = doc.metadata.get("file_type", "")
        
        # 1-indexed page display for PDFs
        display_page = raw_page + 1 if file_type == "pdf" or "page" in doc.metadata else 1
        snippet = doc.page_content.strip()

        # Create unique key based on source filename, page, and snippet prefix
        unique_key = (source_name, display_page, snippet[:100])

        if unique_key not in seen_keys:
            seen_keys.add(unique_key)
            deduped_sources.append({
                "source": source_name,
                "page": display_page,
                "file_type": file_type,
                "snippet": snippet
            })

    return deduped_sources


def get_rag_response(
    llm: Any,
    vector_store: Any,
    question: str,
    k: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
    provider_name: str = "Ollama"
) -> Tuple[str, List[Dict[str, Any]], List[Document]]:
    """
    Execute full RAG pipeline: Retrieve -> Context -> Prompt -> LLM -> Response + Sources.
    
    Args:
        llm: LangChain ChatModel instance.
        vector_store: Active Chroma vector store instance.
        question: User query string.
        k: Top-K documents to retrieve.
        chat_history: Streamlit chat history list.
        provider_name: Name of selected LLM provider for error formatting.
        
    Returns:
        Tuple of (answer_text, deduplicated_sources_list, raw_retrieved_docs).
    """
    if chat_history is None:
        chat_history = []

    # Step 1: Retrieve context chunks
    retrieved_docs = retrieve_context(vector_store, question, k=k)

    # Step 2: Check context availability
    if not retrieved_docs:
        logger.info("No relevant context found in vector store.")
        return NO_CONTEXT_RESPONSE, [], []

    # Step 3: Build context & prompt
    context_str = build_context(retrieved_docs)
    history_str = format_chat_history(chat_history)
    prompt_str = build_prompt(context_str, history_str, question)

    # Step 4: Invoke LLM with error handling
    try:
        response = llm.invoke(prompt_str)
        # Extract response text content from AIMessage or string
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        # Handle empty/null responses
        if not answer_text or not answer_text.strip():
            answer_text = NO_CONTEXT_RESPONSE

    except Exception as e:
        logger.error(f"Error calling LLM provider {provider_name}: {e}")
        error_msg = format_error_message(e, provider=provider_name)
        return error_msg, [], retrieved_docs

    # Step 5: Extract and deduplicate source citations
    sources = extract_deduplicated_sources(retrieved_docs)

    return answer_text, sources, retrieved_docs
