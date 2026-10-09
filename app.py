"""
app.py - Main Streamlit user interface for Local RAG Chatbot 🤖
"""

import os
import streamlit as st
from pathlib import Path

# Set Streamlit page layout early
st.set_page_config(
    page_title="Local RAG Chatbot 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, DEFAULT_TEMPERATURE,
    EMBEDDING_MODEL_NAME, LLM_PROVIDERS, CHROMA_COLLECTION_NAME
)
from utils import (
    setup_environment, initialize_session_state, load_processed_hashes,
    is_valid_file, logger, format_error_message
)
from document_processor import process_uploaded_files
from vector_store import (
    load_embedding_model, load_vector_store, add_documents, reset_vector_store
)
from llm import get_llm
from rag_pipeline import get_rag_response

# Custom UI styling injection for clean, modern aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 10px 14px;
        border-radius: 8px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-top: 6px;
    }
    .metric-badge {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # 1. Environment & State Initialization
    setup_environment()
    initialize_session_state()
    
    # Load embedding model (cached)
    embedding_model = load_embedding_model()
    
    # Load or initialize vector store
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.session_state.vector_store = load_vector_store(embedding_model)

    # Sync processed documents stats from hash store
    hashes_data = load_processed_hashes()
    processed_files_list = list({meta["filename"] for meta in hashes_data.values()})
    st.session_state.processed_files = processed_files_list
    st.session_state.total_chunks = sum(meta.get("chunk_count", 0) for meta in hashes_data.values())

    # =========================================================
    # SIDEBAR: CONTROLS & STATUS
    # =========================================================
    with st.sidebar:
        st.title("🤖 Local RAG Panel")
        
        # System Status Indicator
        st.markdown("### 🚦 Status")
        st.info(f"{st.session_state.status} - {st.session_state.status_message}")

        # Document Statistics Card
        st.markdown("### 📊 Knowledge Base Stats")
        st.markdown(f"""
        - **Documents:** {len(st.session_state.processed_files)}
        - **Total Chunks:** {st.session_state.total_chunks}
        - **Vector Store:** ChromaDB
        - **Embedding:** `all-MiniLM-L6-v2`
        """)
        st.divider()

        # LLM Provider Configuration
        st.markdown("### ⚙️ LLM Configuration")
        provider = st.selectbox(
            "Provider",
            options=list(LLM_PROVIDERS.keys()),
            index=0,
            help="Select local Ollama (no key needed) or cloud providers."
        )

        provider_info = LLM_PROVIDERS[provider]
        selected_model = st.selectbox(
            "Model",
            options=provider_info["models"],
            index=0
        )

        # API Key handling for cloud providers
        if provider_info["requires_api_key"]:
            env_var_name = provider_info["env_var"]
            existing_key = os.getenv(env_var_name, "")
            api_key_input = st.text_input(
                f"{provider} API Key",
                value=existing_key,
                type="password",
                help=f"Loaded from environment variable `{env_var_name}` or entered here."
            )
            if api_key_input:
                os.environ[env_var_name] = api_key_input

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
            help="Lower values are more factual and direct."
        )

        st.divider()

        # Document Upload Section
        st.markdown("### 📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or DOCX",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Supports multiple files."
        )

        st.markdown("### ✂️ Chunking Settings")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=100
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=DEFAULT_CHUNK_OVERLAP,
            step=25
        )

        # Validation rule: Chunk Overlap must be smaller than Chunk Size
        if chunk_overlap >= chunk_size:
            st.error("⚠️ Chunk Overlap must be smaller than Chunk Size!")

        st.markdown("### 🎯 Retrieval Settings")
        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            step=1,
            help="Number of document chunks retrieved for context."
        )

        st.divider()

        # Action Buttons
        st.markdown("### 🚀 Actions")
        
        # PROCESS DOCUMENTS BUTTON
        if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("⚠️ Please select at least one document to upload.")
            elif chunk_overlap >= chunk_size:
                st.error("⚠️ Please fix chunk settings before processing.")
            else:
                st.session_state.status = "🟡 Processing"
                st.session_state.status_message = "Ingesting and embedding documents..."
                
                with st.spinner("Processing documents into vector store..."):
                    try:
                        chunks, newly_processed, skipped_files = process_uploaded_files(
                            uploaded_files, chunk_size, chunk_overlap
                        )

                        if chunks:
                            st.session_state.vector_store = add_documents(
                                st.session_state.vector_store, chunks, embedding_model
                            )
                            # Update hashes and stats
                            updated_hashes = load_processed_hashes()
                            st.session_state.processed_files = list({m["filename"] for m in updated_hashes.values()})
                            st.session_state.total_chunks = sum(m.get("chunk_count", 0) for m in updated_hashes.values())
                            
                            st.session_state.status = "🟢 Ready"
                            st.session_state.status_message = f"Processed {len(newly_processed)} new document(s)."
                            st.success(f"✅ Successfully ingested {len(newly_processed)} file(s) into ChromaDB!")

                        if skipped_files:
                            st.info(f"ℹ️ Skipped {len(skipped_files)} already processed duplicate document(s): {', '.join(skipped_files)}")

                        if not chunks and not skipped_files:
                            st.warning("⚠️ No readable text could be extracted from uploaded files.")
                            st.session_state.status = "🟢 Ready"
                            st.session_state.status_message = "Idle"

                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        st.session_state.status = "🔴 Error"
                        st.session_state.status_message = "Document processing failed."
                        st.error(f"Processing error: {e}")

        # CLEAR CHAT BUTTON
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_retrieved_sources = []
            st.toast("Chat history cleared!", icon="🧹")
            st.rerun()

        # RESET KNOWLEDGE BASE BUTTON
        if st.button("⚠️ Reset Knowledge Base", use_container_width=True):
            with st.spinner("Resetting vector database and hashes..."):
                if reset_vector_store():
                    st.session_state.vector_store = None
                    st.session_state.processed_files = []
                    st.session_state.total_chunks = 0
                    st.session_state.messages = []
                    st.session_state.last_retrieved_sources = []
                    st.session_state.status = "🟢 Ready"
                    st.session_state.status_message = "Knowledge base reset."
                    st.success("✅ Knowledge base reset successfully!")
                    st.rerun()
                else:
                    st.error("Failed to reset knowledge base.")

    # =========================================================
    # MAIN AREA: HEADER & TABS
    # =========================================================
    st.markdown('<div class="main-header">📚 Local RAG Chatbot 🤖</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chat with your documents using AI-powered semantic search.</div>', unsafe_allow_html=True)

    tab_chat, tab_settings, tab_sources = st.tabs(["💬 Chat", "⚙️ System Settings", "🔍 Sources & Output"])

    # ---------------------------------------------------------
    # TAB 1: CHAT INTERFACE
    # ---------------------------------------------------------
    with tab_chat:
        # Check empty state (no documents processed yet)
        if not st.session_state.processed_files:
            st.markdown("""
            ### 📚 Welcome to Local RAG Chatbot
            Upload **PDF**, **TXT**, or **DOCX** files from the sidebar and click **Process Documents** to get started.

            #### 💡 Example Questions You Can Ask:
            - *"What is this document about?"*
            - *"Summarize the main concepts."*
            - *"What are the important key points?"*
            - *"Explain this topic in simple terms."*
            """)
            st.info("👈 Upload your documents in the sidebar to begin querying.")
        else:
            # Display conversation history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # If message has attached sources, render them in expanders
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Sources & Retrieved Snippets", expanded=False):
                            for src in message["sources"]:
                                source_name = src.get("source", "Unknown")
                                page_num = src.get("page", 1)
                                file_type = src.get("file_type", "")
                                snippet = src.get("snippet", "")
                                
                                page_label = f" — Page {page_num}" if file_type == "pdf" else ""
                                st.markdown(f"**📄 {source_name}{page_label}**")
                                st.markdown(f'```text\n{snippet}\n```')

        # User Question Input
        user_query = st.chat_input("Ask a question about your uploaded documents...")

        if user_query:
            if chunk_overlap >= chunk_size:
                st.error("⚠️ Please fix Chunk Overlap settings in the sidebar before asking questions.")
                st.stop()

            # Render User message immediately
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate Assistant Answer
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant context and generating answer..."):
                    try:
                        # Instantiate selected LLM provider
                        llm_instance = get_llm(
                            provider=provider,
                            model_name=selected_model,
                            temperature=temperature
                        )

                        answer, sources, retrieved_docs = get_rag_response(
                            llm=llm_instance,
                            vector_store=st.session_state.vector_store,
                            question=user_query,
                            k=top_k,
                            chat_history=st.session_state.messages[:-1],  # Exclude current query
                            provider_name=provider
                        )

                        # Display Answer
                        st.markdown(answer)

                        # Display Sources Expanders
                        if sources:
                            with st.expander("📚 Sources & Retrieved Snippets", expanded=True):
                                for src in sources:
                                    source_name = src.get("source", "Unknown")
                                    page_num = src.get("page", 1)
                                    file_type = src.get("file_type", "")
                                    snippet = src.get("snippet", "")

                                    page_label = f" — Page {page_num}" if file_type == "pdf" else ""
                                    st.markdown(f"**📄 {source_name}{page_label}**")
                                    st.markdown(f'```text\n{snippet}\n```')

                        # Update session state with message and sources
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        st.session_state.last_retrieved_sources = sources

                    except Exception as e:
                        logger.error(f"Chat generation error: {e}")
                        error_msg = format_error_message(e, provider=provider)
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": []
                        })

    # ---------------------------------------------------------
    # TAB 2: SETTINGS OVERVIEW
    # ---------------------------------------------------------
    with tab_settings:
        st.markdown("### ⚙️ System Configuration Overview")
        st.markdown("This tab displays current active pipeline configurations and model metadata.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏗️ Architecture & Frameworks")
            st.markdown(f"""
            - **Application:** Local RAG Chatbot 🤖
            - **RAG Framework:** LangChain
            - **Vector Database:** ChromaDB (Persistent)
            - **Embedding Model:** `{EMBEDDING_MODEL_NAME}`
            - **Retrieval Search:** Similarity Search
            """)

        with col2:
            st.markdown("#### 🎛️ Active Hyperparameters")
            st.markdown(f"""
            - **Chunk Size:** `{chunk_size}` chars
            - **Chunk Overlap:** `{chunk_overlap}` chars
            - **Top-K Retrieval:** `{top_k}` chunks
            - **Active Provider:** `{provider}`
            - **Active Model:** `{selected_model}`
            - **Temperature:** `{temperature}`
            """)

        st.info("💡 To modify these settings, use the left sidebar controls.")

    # ---------------------------------------------------------
    # TAB 3: SOURCES & DEBUGGING OUTPUT
    # ---------------------------------------------------------
    with tab_sources:
        st.markdown("### 🔍 Knowledge Base Transparency Dashboard")
        st.markdown("Inspect ingested document metrics and inspect the latest retrieved context snippets.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Documents", len(st.session_state.processed_files))
        c2.metric("Total Chunks", st.session_state.total_chunks)
        c3.metric("Vector Store", "ChromaDB")
        c4.metric("Active LLM", f"{provider} ({selected_model})")

        st.divider()

        st.markdown("#### 📁 Processed Files in Knowledge Base")
        if st.session_state.processed_files:
            for f in st.session_state.processed_files:
                st.markdown(f"- 📄 `{f}`")
        else:
            st.write("No files processed yet.")

        st.divider()

        st.markdown("#### 🎯 Latest Retrieved Sources Snippets")
        if st.session_state.last_retrieved_sources:
            for idx, src in enumerate(st.session_state.last_retrieved_sources, start=1):
                page_label = f" (Page {src['page']})" if src.get("file_type") == "pdf" else ""
                st.markdown(f"**Snippet #{idx} - 📄 {src['source']}{page_label}**")
                st.code(src["snippet"], language="text")
        else:
            st.write("No queries asked yet or no retrieved snippets stored for the last query.")


if __name__ == "__main__":
    main()
