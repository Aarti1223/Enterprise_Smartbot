# Final RAG Backend with LangChain + Pinecone + HuggingFace

This is a **Retrieval-Augmented Generation (RAG)** backend that integrates:

- ✅ LangChain framework
- ✅ HuggingFace sentence embeddings
- ✅ Pinecone Vector DB
- ✅ OpenAI GPT-based reasoning
- ✅ SQLite for record storage

---

## Features

- Embed `.docx`/text content using `all-MiniLM-L6-v2`
- Store & query vectorized data via Pinecone
- Use GPT for intelligent document Q&A
- Save extracted records in SQLite and JSON
- Supports chunking + multi-query input

---

## Project Structure

```bash
backend/
│
├── Final_rag.py           # Main RAG execution script
├── backend.py             # Reusable backend utilities
├── app.py                 # Flask/streamlit (if used)
├── requirements.txt
├── .env                   # Environment variables (not committed)
│
├── chunks.pkl             # Embedded content chunks
├── combined_data.json     # Combined processed data
├── embedded_chunks.json   # Embedding reference
├── token_usage_log.json   # Usage log (if using OpenAI metering)
├── travel_records.json    # Extracted records (example)
├── upload_done.flag       # Upload tracking
├── users.db               # SQLite DB
└── ...
```
