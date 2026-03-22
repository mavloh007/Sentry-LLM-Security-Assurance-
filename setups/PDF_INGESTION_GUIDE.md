# PDF Ingestion to Supabase Vector Store

## Overview

Your updated `ingest.py` now ingests PDFs directly into **Supabase pgvector** instead of ChromaDB.

## How It Works

```
PDFs in src/documents/
    ↓
PyPDF reads + extracts text
    ↓
Text splits into 800-char chunks (100-char overlap)
    ↓
sentence-transformers generates 384-dim embeddings
    ↓
Each chunk inserted into Supabase documents table with:
  - content (text)
  - embedding (vector)
  - source (filename)
  - metadata (doc_id, chunk_index, etc.)
    ↓
Automatically indexed with pgvector IVFFLAT
```

## Quick Start

### 1. Prepare Your PDFs

Place all PDF files in:
```
src/documents/
├── sgbank_withdrawal_policy_and_procedures.pdf
├── sgbank_emergency_withdrawal_policy.pdf
├── sgbank_identity_verification_and_authentication_policy.pdf
└── sgbank_transaction_monitoring_and_fraud_detection_policy.pdf
```

### 2. Ensure Dependencies

```bash
pip install pypdf sentence-transformers
```

### 3. Run Ingestion

```bash
python ingest.py
```

## Expected Output

```
======================================================================
🚀 STARTING PDF INGESTION TO SUPABASE VECTOR STORE
======================================================================

📄 Found 4 PDFs

🔌 Connecting to Supabase...
✓ Connected to Supabase

📦 Loading embedding model (all-MiniLM-L6-v2)...
✓ Embedding model ready

📥 Processing: sgbank_withdrawal_policy_and_procedures.pdf (doc_id: sgbank_withdrawal_policy_and_procedures)
   ✓ Loaded 45230 characters
   ✓ Split into 58 chunks
      → Ingested 5/58 chunks
      → Ingested 10/58 chunks
      ...
   ✓ Successfully ingested 58 chunks

📥 Processing: sgbank_emergency_withdrawal_policy.pdf (doc_id: sgbank_emergency_withdrawal_policy)
   ✓ Loaded 12340 characters
   ✓ Split into 16 chunks
   ✓ Successfully ingested 16 chunks

...

======================================================================
✅ INGESTION COMPLETE!
======================================================================
Total chunks ingested: 150
Errors encountered: 0

🧪 Testing vector search...

Query: 'withdrawal policy'
  1. [97.53%] Withdrawal policies govern the procedures for...(sgbank_withdrawal_policy_and_procedures)
  2. [87.12%] Standard withdrawal processing limits...(sgbank_withdrawal_policy_and_procedures)

Query: 'emergency withdrawal'
  1. [99.21%] Emergency withdrawals can be processed...(sgbank_emergency_withdrawal_policy)
  2. [85.42%] Contact the emergency line at...(sgbank_emergency_withdrawal_policy)

Query: 'identity verification'
  1. [94.87%] Identity verification requires government-issued...(sgbank_identity_verification_and_authentication_policy)
  2. [82.15%] Acceptable documents include passport...(sgbank_identity_verification_and_authentication_policy)
```

## Key Changes from ChromaDB Version

| Aspect | ChromaDB (Old) | Supabase (New) |
|--------|----------------|----------------|
| **Storage** | Local SQLite file | Cloud PostgreSQL |
| **Embeddings** | Implicit | Explicit (sentence-transformers) |
| **Chunks** | Stored with metadata in ChromaDB | Stored with full metadata in Supabase |
| **Search** | Chroma's cosine similarity | pgvector cosine similarity |
| **Scalability** | Limited to machine storage | Cloud database (scalable) |
| **Multi-user** | Single user only | Per-user access via RLS |
| **Backup** | Manual file copy | Supabase automated backups |

## What Gets Stored in Supabase

For each chunk, these fields are stored:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "content": "Withdrawal policies govern... [chunk text]",
  "embedding": [0.123, 0.456, ..., -0.789],  // 384-dimensional vector
  "source": "sgbank_withdrawal_policy_and_procedures",
  "doc_type": "policy",
  "metadata": {
    "source_filename": "sgbank_withdrawal_policy_and_procedures.pdf",
    "chunk_index": 5,
    "total_chunks": 58,
    "original_doc_id": "sgbank_withdrawal_policy_and_procedures"
  },
  "created_at": "2026-03-20T10:30:45.123456+00:00"
}
```

## How Your Chatbot Uses It

When a user asks a question:

```python
from src.db.supabase_client import SupabaseDB
from sentence_transformers import SentenceTransformer

db = SupabaseDB()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

user_query = "What's the daily withdrawal limit?"
query_embedding = embedder.encode(user_query).tolist()

# Retrieve relevant policy documents
relevant_docs = db.search_documents(
    embedding=query_embedding,
    limit=5,
    threshold=0.7
)

# Use these docs in your LangChain prompt
context = "\n".join([doc['content'] for doc in relevant_docs])
response = llm.invoke(f"Based on this policy:\n{context}\n\nAnswer: {user_query}")
```

## Customization Options

### Change Chunk Size

```bash
# Edit ingest.py, line ~140
chunks = split_text(text, chunk_size=1000, overlap=150)  # Larger chunks
```

### Change Embedding Model

```bash
# Edit ingest.py, line ~105
embedder = SentenceTransformer('all-mpnet-base-v2')  # More powerful but slower
```

### Filter by Document Type

```bash
# In your chatbot code
policy_docs = db.search_documents(embedding, limit=5)
policy_docs = [d for d in policy_docs if d['doc_type'] == 'policy']
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No PDFs found in src/documents/" | Check PDF files exist and have .pdf extension |
| "Connection refused" | Verify SUPABASE_URL & SUPABASE_SERVICE_KEY in .env |
| "Embedding dimension mismatch" | all-MiniLM-L6-v2 must be 384-dim, check encode output |
| "Vector search returns no results" | Wait for pgvector index to build, or lower threshold to 0.5 |
| "Duplicate chunks" | Old chunks still in DB? Clear and re-ingest or migrate_to_supabase.py |

## Performance Tips

✅ **For First Ingestion**
- Small PDFs (< 100 pages) process in seconds
- Large PDFs (> 500 pages) may take 2-5 minutes
- Embedding generation is the bottleneck (CPU-bound)

✅ **For Vector Search**
- First query: slower (index building)
- Subsequent queries: < 100ms with pgvector IVFFLAT index
- Similarity threshold 0.7 recommended for high recall

✅ **For Adding New PDFs**
- Just place new PDF in `src/documents/` and run `python ingest.py` again
- Duplicates won't be created (different file = different IDs)
- Always tests vector search after ingestion

## Re-ingesting Files

If you update PDFs or need to reload:

```bash
# Option 1: Just re-run (creates duplicates - not ideal)
python ingest.py

# Option 2: Clear documents and re-ingest (better)
# In Supabase SQL:
DELETE FROM documents WHERE doc_type = 'policy';

# Then:
python ingest.py
```

## Next: Use in Your Chatbot

Once ingested, integrate into your withdrawal_graph.py:

```python
from src.db.supabase_client import SupabaseDB
from sentence_transformers import SentenceTransformer

class WithdrawalGraphState(TypedDict):
    user_query: str
    retrieved_docs: List[Dict]

async def retrieve_documents(state):
    """Node: Retrieve relevant policy documents"""
    db = SupabaseDB()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = embedder.encode(state['user_query']).tolist()
    docs = db.search_documents(query_embedding, limit=5)
    
    return {
        **state,
        "retrieved_docs": docs
    }
```

Done! Your PDFs are now in the cloud vector database! 🚀
