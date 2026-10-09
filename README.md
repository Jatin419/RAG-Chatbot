# Local RAG Chatbot 🤖

> **"Ask questions from your documents using Retrieval-Augmented Generation."**

An intermediate-level, production-minded **Local Retrieval-Augmented Generation (RAG) Chatbot** built with Python, Streamlit, LangChain, ChromaDB, and local HuggingFace embeddings. Runs 100% locally by default using **Ollama** (`llama3.2`) without requiring cloud API keys, with optional integration for cloud providers like **OpenAI** and **Groq**.

---

## 🌟 Features

- 🔒 **Local-First Privacy**: Embeddings run locally via HuggingFace `sentence-transformers`, vector storage runs locally via `ChromaDB`, and inference runs locally via `Ollama`. No API key required for standard operation.
- 📄 **Multi-Format Document Support**: Seamlessly ingest, extract text from, and chunk `.pdf`, `.txt`, and `.docx` files.
- ⚡ **SHA-256 Duplicate Detection**: Prevents redundant vector processing and indexing by tracking file content hashes in a lightweight JSON store.
- 🧠 **Conversational Memory**: Retains multi-turn conversation history so the LLM understands context in follow-up queries.
- 📌 **Exact Source Attribution & Deduplication**: Every generated answer displays exact source document filenames, 1-indexed page numbers (for PDFs), and snippet expanders. Duplicate source chunks are automatically deduplicated.
- 🚫 **Strict Grounding / Anti-Hallucination**: Enforces a strict system prompt instructing the model to reply *only* using retrieved context. Returns a friendly fallback response when information is not present in uploaded documents.
- 🎛️ **Configurable RAG Hyperparameters**: Interactive sidebar controls for Chunk Size (500–2000), Chunk Overlap (0–500), Top-K retrieval (1–10), and LLM Temperature (0.0–1.0).
- 🧹 **State Management**: Dedicated actions to **Clear Chat History** (persisting vectors) or **Reset Knowledge Base** (wiping ChromaDB & hashes).
- 🔍 **Transparency Dashboard**: Includes dedicated **Settings** and **Sources & Output** tabs for monitoring system metrics, chunk counts, and raw retrieved context.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **Frontend UI** | Streamlit |
| **RAG Orchestration** | LangChain (`langchain-core`, `langchain-community`) |
| **Document Loaders** | `PyPDFLoader`, `TextLoader`, `Docx2txtLoader` |
| **Text Splitter** | `RecursiveCharacterTextSplitter` |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Database** | ChromaDB (Persistent Disk Store) |
| **Primary LLM Provider** | Ollama (`llama3.2`) |
| **Optional LLM Providers**| OpenAI (`gpt-4o-mini`), Groq (`llama-3.3-70b-versatile`) |
| **Environment Management**| `python-dotenv` |

---

## 🏗️ System Architecture

```text
                    DOCUMENTS
                PDF / TXT / DOCX
                       │
                       ▼
              ┌─────────────────┐
              │ Document Loader │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Text Splitter   │
              │ Recursive       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ HuggingFace     │
              │ Embeddings      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    ChromaDB     │
              │ Vector Database  │
              └────────┬────────┘
                       │
                 USER QUESTION
                       │
                       ▼
              ┌─────────────────┐
              │ Similarity      │
              │ Search          │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Relevant        │
              │ Context         │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ LLM             │
              │ Ollama/OpenAI/  │
              │ Groq            │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Answer +        │
              │ Source Snippets │
              └─────────────────┘
```

---

## 📁 Project Structure

```text
local-rag-chatbot/
│
├── app.py                  # Streamlit UI, session state, tabs, sidebar, user workflow
├── config.py               # Application constants, paths, default hyperparameters, provider schemas
├── document_processor.py   # Document loading (PDF/TXT/DOCX), splitting, hash generation, duplicate check
├── vector_store.py         # Embedding model loading (cached), ChromaDB creation/loading/retrieval/reset
├── rag_pipeline.py         # Retrieval, context assembly, prompt template formatting, LLM call, source dedup
├── llm.py                  # Dynamic LLM provider factory (Ollama, OpenAI, Groq)
├── utils.py                # Logging configuration, file validation, error formatting, hash storage
│
├── requirements.txt        # Production dependencies
├── .env.example            # Sample environment file template
├── .gitignore              # Git exclusion rules
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone & Setup Virtual Environment

#### **Windows (PowerShell)**:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### **macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🤖 LLM Provider Setup

### Option A: Local Ollama (Default - Primary Local Mode)

1. Download and install [Ollama](https://ollama.ai/).
2. Pull the default lightweight model:
   ```bash
   ollama pull llama3.2
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
4. Run the Streamlit app. No API keys are required!

---

### Option B: OpenAI Setup (Optional Cloud)

1. Create a `.env` file in the root directory by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```
2. Add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-proj-xxxx...
   ```
3. In the sidebar of the app, select **Provider: OpenAI** and **Model: gpt-4o-mini**.

---

### Option C: Groq Setup (Optional Cloud)

1. Add your Groq API key to `.env`:
   ```env
   GROQ_API_KEY=gsk_xxxx...
   ```
2. In the sidebar of the app, select **Provider: Groq** and **Model: llama-3.3-70b-versatile**.

---

## 🚀 Running the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔄 RAG Data Flow & Workflow

### 1. Document Ingestion Flow
1. **Upload & Validation**: File type checked (`.pdf`, `.txt`, `.docx`).
2. **SHA-256 Hashing**: File content hashed. If the hash exists in `processed_hashes.json`, ingestion is skipped with an informative message.
3. **Parsing**: Loaded using PyPDFLoader, TextLoader, or Docx2txtLoader preserving metadata (`source`, `page`, `file_type`).
4. **Recursive Chunking**: Split into chunks using `RecursiveCharacterTextSplitter` (default 1000 size, 150 overlap).
5. **Embedding & Storage**: Vectorized using HuggingFace `all-MiniLM-L6-v2` and stored in persistent ChromaDB (`./chroma_db`).

### 2. Question Answering Query Flow
1. **User Question**: Question submitted via Streamlit chat input.
2. **Similarity Search**: Queries ChromaDB vector store for top-K matching chunks.
3. **Context & History Assembly**: Concatenates retrieved document chunks into a structured context block along with recent chat history.
4. **Prompt Engineering**: Formats context and query into a strict non-hallucinating system prompt.
5. **LLM Generation**: Model generates grounded response using context.
6. **Citation & Display**: Returns answer along with expandable source snippets containing document filename and 1-indexed page number.

---

## 💬 Example Scenarios & Test Questions

### Scenario 1: Multi-Document Comparison
**Files Uploaded**: `machine_learning.pdf` and `deep_learning.pdf`  
**User Question**: *"Compare machine learning and deep learning based on the uploaded documents."*  
**Expected Result**: System retrieves matching chunks from both files, formats a comparative summary, and lists both files in the **Sources** expander.

### Scenario 2: Follow-Up Questions (Conversational Memory)
- **Question 1**: *"What is supervised learning?"*
- **Question 2**: *"What are its main advantages?"*
- **Question 3**: *"Give me a practical example of it."*  
**Expected Result**: The assistant understands that "its" and "it" refer to supervised learning using chat history.

### Scenario 3: Out-of-Domain Query (No Context Fallback)
**User Question**: *"What is the capital of France?"*  
**Expected Result**:  
> *"I couldn't find relevant information in the uploaded documents. Please try rephrasing your question or upload a document containing the required information."*

---

## 🧪 Comprehensive Verification Checklist

The codebase includes handling for all core production edge cases:
1. ✅ **Application Startup**: Runs cleanly without error.
2. ✅ **PDF Processing**: Retains page numbers accurately.
3. ✅ **TXT Processing**: Correctly handles text encoding.
4. ✅ **DOCX Processing**: Extracts formatted paragraphs cleanly.
5. ✅ **Multi-Document RAG**: Merges context from diverse files.
6. ✅ **SHA-256 Duplicate Detection**: Skips re-embedding identical files.
7. ✅ **Vector Database Persistence**: Retains embeddings across app restarts.
8. ✅ **Empty File Protection**: Prevents indexing whitespace-only files.
9. ✅ **Unsupported File Rejection**: Displays friendly warning for disallowed file types.
10. ✅ **Invalid Chunk Overlap**: Prevents execution when `chunk_overlap >= chunk_size`.
11. ✅ **Ollama Unavailable Graceful Fallback**: Shows connection instructions if `ollama serve` is down.
12. ✅ **Missing API Key Validation**: Notifies user cleanly if OpenAI/Groq keys are missing.
13. ✅ **Anti-Hallucination Fallback**: Returns explicit fallback message when context is absent.
14. ✅ **Source Attribution**: Deduplicates and displays exact text snippets with file and page numbers.
15. ✅ **Clear Chat**: Deletes message history without deleting vector index.
16. ✅ **Reset Knowledge Base**: Safely deletes ChromaDB directory and resets hash registry.

---

## 📈 Portfolio Scaling Path (MVP -> Enterprise Production)

This project is intentionally designed as a **clean, intermediate portfolio MVP**. Here is how to explain scaling it to enterprise-grade production during technical interviews:

| Component | Portfolio MVP (Current) | Production Enterprise Architecture |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | React / Next.js / TailwindCSS |
| **Backend API** | Embedded Python script | FastAPI / Async Python microservices |
| **Vector DB** | Local ChromaDB | Qdrant / Pinecone / PostgreSQL + `pgvector` |
| **Embedding Engine**| HuggingFace CPU | vLLM / NVIDIA Triton / OpenAI `text-embedding-3-small` |
| **Task Queue** | Synchronous Streamlit state | Celery + Redis / Apache Kafka |
| **RAG Evaluation** | Manual inspection | RAGAS framework / LangSmith tracing |
| **Reranking** | Top-K Similarity | Cross-Encoder Reranker (Cohere / BGE-Reranker) |
| **Deployment** | Localhost | Docker + Kubernetes on AWS/GCP |

---

## 🎯 Resume Bullet Points

- **Developed a Local RAG Chatbot** using Streamlit, LangChain, ChromaDB, HuggingFace embeddings (`all-MiniLM-L6-v2`), and Ollama/OpenAI/Groq for private document-grounded QA.
- **Architected Multi-Format Document Ingestion Pipeline** supporting PDF, TXT, and DOCX files with SHA-256 duplicate detection, configurable recursive text splitting, and persistent vector indexing.
- **Engineered Context-Aware QA Engine** featuring top-K similarity retrieval, strict anti-hallucination prompt guardrails, conversational memory, and source attribution with page-level citations.

---

## ❓ Technical Interview Q&A Guide

<details>
<summary><b>Q1: What is Retrieval-Augmented Generation (RAG)?</b></summary>
RAG is an architectural pattern that enhances Large Language Models by retrieving relevant knowledge chunks from an external database (vector store) and feeding them into the LLM prompt as context. This ensures factual grounding and reduces hallucinations without costly model fine-tuning.
</details>

<details>
<summary><b>Q2: Why choose RAG over Fine-Tuning?</b></summary>
Fine-tuning updates model weights for style or domain jargon but does not reliably store specific facts or source citations. RAG allows real-time dynamic updates to knowledge bases simply by uploading/deleting documents, provides exact source attribution, and preserves privacy.
</details>

<details>
<summary><b>Q3: What are vector embeddings?</b></summary>
Embeddings are high-dimensional numerical vector representations of text strings. Words or passages with similar semantic meanings are placed close together in the vector space, allowing semantic similarity search beyond simple keyword matching.
</details>

<details>
<summary><b>Q4: Why use sentence-transformers/all-MiniLM-L6-v2?</b></summary>
It is an extremely lightweight (80MB), fast, and effective embedding model generating 384-dimensional vectors. It runs efficiently on CPU without requiring GPU hardware, making it ideal for local RAG applications.
</details>

<details>
<summary><b>Q5: What is ChromaDB and why use it here?</b></summary>
ChromaDB is an open-source vector database designed for AI applications. It offers zero-setup local persistence on disk, fast similarity search, and seamless native integration with LangChain.
</details>

<details>
<summary><b>Q6: What is chunking and why is RecursiveCharacterTextSplitter used?</b></summary>
Chunking breaks large documents into smaller text blocks that fit within embedding and LLM context windows. `RecursiveCharacterTextSplitter` tries to split by paragraphs (`\n\n`), lines (`\n`), sentences (`. `), and words (` `) sequentially, preserving natural language boundaries.
</details>

<details>
<summary><b>Q7: What is Top-K retrieval?</b></summary>
Top-K represents the number of most semantically similar document chunks retrieved from the vector database for a given user query. Higher K provides more context but increases prompt length and processing cost.
</details>

<details>
<summary><b>Q8: How does conversational memory work in this app?</b></summary>
Recent chat messages stored in Streamlit's `st.session_state` are formatted into a conversational history string and passed into the RAG prompt template, enabling the model to resolve coreferences (e.g., "What are *its* benefits?").
</details>

<details>
<summary><b>Q9: How does the system prevent hallucinations?</b></summary>
The prompt explicitly instructs the LLM to rely *only* on the provided context. If no matching chunks are retrieved or if the answer is not present in the context, the system returns a pre-defined fallback message rather than generating unsupported facts.
</details>

<details>
<summary><b>Q10: Why use Ollama by default?</b></summary>
Ollama enables running quantized open-weights models (like Llama 3.2) locally on consumer hardware without sending data over the internet or requiring paid cloud API keys.
</details>

<details>
<summary><b>Q11: How is duplicate document detection implemented?</b></summary>
Before processing an uploaded file, its raw byte content is hashed using SHA-256. The hash is checked against a local JSON metadata store. If the hash exists, processing is skipped.
</details>

<details>
<summary><b>Q12: How are source citations extracted and deduplicated?</b></summary>
When chunks are retrieved, their metadata (`source`, `page`, `file_type`, `snippet`) is collected. Unique combinations of filename, page number, and snippet prefix are filtered to prevent redundant source expanders.
</details>

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
