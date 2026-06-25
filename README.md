```markdown
# 📚 Modular Multimodal RAG Engine
> **A High-Performance, Database-Agnostic Intelligence Layer for Document Reasoning.**

The **Modular RAG Engine** is a professional-grade Python library that enables developers to build sophisticated AI systems capable of reading, remembering, and reasoning over complex, unstructured data. 

Unlike standard RAG libraries, this engine is built for **Production Stability**:
*   **Multi-Modal Vision:** Understands charts, tables, and diagrams.
*   **Database Agnostic:** Works with PostgreSQL (Supabase), SQLite, MySQL, etc., via SQLAlchemy.
*   **Provider Agnostic:** Inject any LLM (OpenAI, Gemini, Groq, Anthropic).
*   **Multi-Tenant:** Built-in isolation ensures User A never sees User B's data.
*   **Atomic Ingestion:** Fingerprinting (SHA-256) prevents duplicate costs and data corruption.

---

## 🛠 Prerequisites

To run this engine, you must provide your own "Fuel" (API Keys) from the following providers:

| Provider | Purpose |
| :--- | :--- |
| **LlamaCloud** | Advanced Multimodal Document Parsing (Vision). |
| **Pinecone** | Serverless Vector Database for semantic search. |
| **Any SQL DB** | Metadata storage (Supports Supabase/Postgres, SQLite, etc). |
| **Google/Groq** | Inference models for planning and answering. |

---

## 🚀 Installation

Install via terminal:
```bash
pip install git+https://github.com/rajpratik144/modular-rag-engine.git
```

---

## 📖 API Reference: The "Universal Storage" Concept

The system uses **SQLAlchemy**. You no longer need specific SDKs for Supabase. You simply provide a standard `DATABASE_URL`.

### 1. The Configuration Object
Every method requires a `config` dictionary.

| Key | Required | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | **Yes** | SQLAlchemy string (e.g. `postgresql://...` or `sqlite:///./db.db`) |
| `LLAMA_CLOUD_API_KEY` | Yes | API Key for document parsing. |
| `PINECONE_API_KEY` | Yes | Your Pinecone API Key. |
| `PINECONE_INDEX_NAME`| Yes | The name of your 768-dim index. |
| `GOOGLE_API_KEY` | Yes* | Required if using Google Gemini Embeddings. |
| `VISION_MODEL` | No | Default: `openai-gpt-4o-mini`. |

### 2. Core Methods

*   `RAGCoreEngine(config, planner_llm, brain_llm, system_persona=None)`: Initializes the engine.
*   `.ingest(file_paths, user_id)`: Fingerprints, parses, embeds, and stores documents.
*   `.ask(question, user_id, chat_history=None)`: Performs multi-query search and returns a response.
*   `.delete_files(doc_id, user_id)`: Surgically removes data from SQL and Vector databases.

---

## 🤖 Example: Building a "Legal Analyst" Bot

```python
import os
from langchain_groq import ChatGroq
from rag_engine.core.engine import RAGCoreEngine

# 1. SETUP CONFIG (Universal SQL)
config = {
    "DATABASE_URL": "sqlite:///./legal_bot.db", # Or your Supabase/Postgres string
    "LLAMA_CLOUD_API_KEY": "llx-...",
    "PINECONE_API_KEY": "pcsk_...",
    "PINECONE_INDEX_NAME": "legal-index",
    "GROQ_API_KEY": "gsk_...",
    "VISION_MODEL": "openai-gpt-4o-mini"
}

# 2. CHOOSE THE BRAIN
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=config["GROQ_API_KEY"])

# 3. INITIALIZE
engine = RAGCoreEngine(config, planner_llm=llm, brain_llm=llm)

# 4. INGESTION
user_id = "lawyer_pro_1"
engine.ingest(["./case_file.pdf"], user_id)

# 5. QUERYING
answer = engine.ask("Summarize the key arguments.", user_id)
print(f"Assistant: {answer}")
```

---

## 🏗 Architecture
1.  **`parsers`**: The "Eyes." Converts visual documents into structured Markdown.
2.  **`storage`**: The "Vault." Manages SQL metadata via SQLAlchemy and Vector data via Pinecone.
3.  **`core`**: The "Brain." Implements Query Expansion and Context-Aware reasoning.

---

**Author:** Pratik Raj  
**Version:** 0.1.1 (Universal SQL Update)
```
