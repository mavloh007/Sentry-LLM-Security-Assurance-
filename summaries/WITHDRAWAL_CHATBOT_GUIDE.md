# SGBank Withdrawal Chatbot - Complete Guide

## Overview

The SGBank Withdrawal Chatbot is a **multi-agent, document-scoped RAG (Retrieval Augmented Generation) system** that helps customers with questions about withdrawal policies. It uses:

- **LangChain + LangGraph** for multi-agent orchestration
- **Supabase PostgreSQL + pgvector** for persistent data storage and vector similarity search
- **Sentence Transformers** (all-MiniLM-L6-v2) for 384-dimensional embeddings
- **OpenAI GPT-4o-mini** for language generation
- **Sentinel Guard** for input/output safety validation

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input (CLI)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │   WithdrawalChatbot (main.py)           │
        │  - Routes queries to appropriate agent  │
        │  - Validates input/output with Sentinel │
        │  - Stores all messages to Supabase      │
        └────┬────────────────────┬───────────────┘
             │                    │
             ▼                    ▼
    ┌──────────────────┐  ┌──────────────────┐
    │  4 Policy Agents │  │  SupabaseDB      │
    │  - Withdrawal    │  │  - Users         │
    │  - Emergency     │  │  - Conversations │
    │  - Identity      │  │  - Messages      │
    │  - Fraud         │  │  - Documents     │
    └────────┬─────────┘  │  - Audit Logs    │
             │            │  - Message Flags │
             ▼            └──────────────────┘
    ┌──────────────────┐            │
    │   RAG Tool       │            │
    │  - Generate      │            ▼
    │    embeddings    │    ┌──────────────────┐
    │  - Search Supabase   │   Supabase Cloud │
    │  - Retrieve docs │    │  PostgreSQL +    │
    └──────────────────┘    │  pgvector (82    │
                            │  documents)      │
                            └──────────────────┘
```

### Agent Routing

The chatbot automatically routes queries to the appropriate agent based on keywords:

| Agent | Triggered By | Purpose |
|-------|--------------|---------|
| **Withdrawal** | Default (withdrawal limits, processing times, channels) | Standard withdrawal policies and procedures |
| **Emergency** | "emergency", "urgent", "medical", "hospital", "bereavement" | Emergency withdrawal policies |
| **Identity** | "verify", "authenticate", "kyc", "otp", "biometric", "proof of identity" | Identity verification and authentication |
| **Fraud** | "monitoring", "flagged", "blocked", "aml", "suspicious", "fraud" | Transaction monitoring and fraud detection |

## Data Flow

### 1. **User Message**
```
User Input → Stored in Supabase (messages table) → Routed to agent
```

### 2. **Retrieval & Generation**
```
Query → Generate Embedding (Sentence Transformer)
      → Search Supabase pgvector (cosine similarity)
      → Retrieve top-K relevant documents
      → Pass to LLM with system prompt
      → Generate response
```

### 3. **Safety Validation**
```
Input → Sentinel Guard → Blocked? → Flag message + Return refusal
         ↓                  ↓
       Allowed         Continue
         ↓
      LLM Response → Sentinel Guard → Blocked? → Flag + Return refusal
                                         ↓
                                     Allowed
                                         ↓
                                    Return response
```

### 4. **Persistence**
```
All interactions → Stored in Supabase:
  ├─ messages (user/assistant)
  ├─ audit_logs (action tracking)
  ├─ message_flags (suspicious content)
  └─ conversations (session grouping)
```

## Key Features

### ✅ Multi-Agent Routing
- Automatic query classification based on keywords
- Each agent trained on specific policy documents
- Prevents hallucination outside document scope

### ✅ Document-Scoped RAG
- Each agent searches only its approved documents
- 82 policy documents ingested with pgvector embeddings
- Cosine similarity search with configurable threshold

### ✅ Persistent Storage
- All messages stored in Supabase (no data loss on restart)
- Conversation history maintained per session
- Audit trail of all actions and errors

### ✅ Security & Safety
- Input validation with Sentinel Guard
- Output validation to prevent policy violations
- Suspicious content flagging
- Row-level security for multi-user support

### ✅ User Tracking
- Fixed test user: `local-test-user` (UUID: stable across runs)
- Conversation management per user
- Audit logs with timestamps

## File Structure

```
src/chatbot/
├── withdrawal_chatbot.py          # Main chatbot class & agents
└── sentinel_guard.py               # Safety validation

src/db/
├── supabase_client.py              # Database API wrapper (482 lines)
└── __init__.py                     # Package exports

src/documents/                       # 4 policy PDF files (82 chunks ingested)
├── sgbank_withdrawal_policy_and_procedures.pdf
├── sgbank_emergency_withdrawal_policy.pdf
├── sgbank_identity_verification_and_authentication_policy.pdf
└── sgbank_transaction_monitoring_and_fraud_detection_policy.pdf

main.py                             # CLI entry point
ingest.py                           # PDF → Supabase ingestion pipeline
```

## Setup & Usage

### 1. **Prerequisites**

Ensure you have:
- Supabase project created with pgvector enabled
- `.env` file with Supabase credentials:
  ```env
  SUPABASE_URL=https://your-project.supabase.co
  SUPABASE_SERVICE_KEY=your-service-role-key
  OPENAI_API_KEY=your-openai-api-key
  SENTINEL_API_KEY=your-sentinel-api-key (optional)
  ```

### 2. **Database Schema**

Run this SQL in your Supabase SQL Editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversations table
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents table (384-dim embeddings)
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT NOT NULL,
  embedding vector(384),
  source TEXT,
  doc_type TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for vector search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Audit logs table
CREATE TABLE audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  action TEXT NOT NULL,
  resource_type TEXT,
  details JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Message flags table (suspicious content)
CREATE TABLE message_flags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  reason TEXT NOT NULL,
  details JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP
);
```

### 3. **Ingest Documents**

Run the ingestion pipeline:

```bash
python ingest.py
```

This will:
- Scan `src/documents/` for PDF files
- Extract text and split into 800-char chunks (100-char overlap)
- Generate embeddings using sentence-transformers
- Insert into Supabase with metadata

### 4. **Run the Chatbot**

```bash
python main.py
```

Output:
```
Connected to Supabase database.
[DEBUG] Total documents in database: 82
[DEBUG] Document sources in database:
  → sgbank_withdrawal_policy_and_procedures: 18 documents
  → sgbank_emergency_withdrawal_policy: 8 documents
  → sgbank_identity_verification_and_authentication_policy: 2 documents
  → sgbank_transaction_monitoring_and_fraud_detection_policy: 54 documents

SGBank Withdrawal Assistant Ready.
Type 'exit' to quit.

You: how do I withdraw money?
```

## Example Interactions

### Query 1: Standard Withdrawal
```
You: how do I withdraw money?

[RAG TOOL DEBUG] for doc_id: sgbank_withdrawal_policy_and_procedures
[RAG TOOL DEBUG] Query: 'how do I withdraw money?'
[RAG TOOL DEBUG] Total results from search: 3
[RAG TOOL DEBUG] After filtering for 'sgbank_withdrawal_policy_and_procedures': 3 docs