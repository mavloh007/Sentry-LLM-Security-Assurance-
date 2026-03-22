# Withdrawal Chatbot - Supabase Integration Summary

## What Changed?

The withdrawal chatbot was refactored from a **local ChromaDB system** to a **cloud-based Supabase + pgvector system** with persistent storage, user tracking, and audit logging.

### Before ❌
- Used local ChromaDB vector store
- No persistent storage (lost on restart)
- Single session, no multi-user support
- No audit trail
- No message history across sessions

### After ✅
- Uses Supabase PostgreSQL with pgvector
- All messages persist in cloud
- Multi-user ready with user tracking
- Complete audit logging of all actions
- Full conversation history per user
- Security flagging for suspicious content

## Architecture

```
User Input (CLI)
    ↓
main.py
    ├─ Connect to Supabase
    └─ Initialize chatbot
    ↓
WithdrawalChatbot.chat()
    ├─ Store user message
    ├─ Route to agent
    ├─ Validate with Sentinel
    ├─ Call RAG tool
    ├─ Generate response
    ├─ Validate output
    ├─ Store response
    └─ Log audit trail
    ↓
SupabaseDB (API wrapper)
    ├─ add_message()
    ├─ search_documents()
    ├─ create_audit_log()
    └─ flag_message_as_suspicious()
    ↓
Supabase Cloud
    ├─ PostgreSQL database
    ├─ pgvector extension
    └─ 7 tables (users, conversations, messages, documents, audit_logs, etc.)
```

## Key Files Modified

### 1. **main.py** ✅ Updated
- Changed from ChromaDB to SupabaseDB
- Added health check before chatting
- Simplified initialization

### 2. **src/chatbot/withdrawal_chatbot.py** ✅ Updated
- Imports: `from src.db.supabase_client import SupabaseDB`
- Constructor: Takes `db` parameter instead of `vector_store`
- User: Fixed to `local-test-user` (UUID) for local testing
- Chat method: Now stores all messages to Supabase
  - Stores user message before processing
  - Creates audit logs
  - Flags suspicious content
  - Stores assistant response

### 3. **src/db/supabase_client.py** ✅ Created (482 lines)
Complete database API wrapper with:
- User management (`create_user`, `get_user`)
- Conversation tracking (`create_conversation`, `list_user_conversations`)
- Message persistence (`add_message`, `get_conversation_history`)
- Vector search (`search_documents` with fallback)
- Audit logging (`create_audit_log`)
- Security flagging (`flag_message_as_suspicious`)

### 4. **ingest.py** ✅ Updated
- Changed from ChromaDB to Supabase ingestion
- Generates embeddings locally
- Stores to Supabase documents table
- Now requires Supabase connection

## Database Tables

7 tables created in Supabase:

```sql
-- User profiles
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email TEXT,
  metadata JSONB
)

-- Chat sessions
CREATE TABLE conversations (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  title TEXT
)

-- Chat history
CREATE TABLE messages (
  id UUID PRIMARY KEY,
  conversation_id UUID REFERENCES conversations(id),
  user_id UUID REFERENCES users(id),
  role TEXT ('user' or 'assistant'),
  content TEXT
)

-- Policy documents (with embeddings)
CREATE TABLE documents (
  id UUID PRIMARY KEY,
  content TEXT,
  embedding vector(384),  -- pgvector
  source TEXT,
  doc_type TEXT,
  metadata JSONB
)

-- Action tracking
CREATE TABLE audit_logs (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  action TEXT,
  resource_type TEXT,
  details JSONB
)

-- Suspicious content tracking
CREATE TABLE message_flags (
  id UUID PRIMARY KEY,
  message_id UUID REFERENCES messages(id),
  reason TEXT,
  details JSONB
)

-- Session management
CREATE TABLE sessions (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  expires_at TIMESTAMP
)
```

## How It Works - Step by Step

### 1. User asks a question
```
You: "How do I withdraw money?"
```

### 2. Chatbot stores the message
```python
db.add_message(
    conversation_id=self.conversation_id,
    user_id=self.user_id,
    role="user",
    content="How do I withdraw money?",
    metadata={"routed_to": "withdrawal"}
)
```

### 3. Chatbot creates audit log
```python
db.create_audit_log(
    user_id=self.user_id,
    action="message_received",
    resource="conversation",
    details={"agent": "withdrawal"}
)
```

### 4. Sentinel Guard validates input
```python
if self._check_sentinel_input(agent_key, user_message):
    # Block and flag
    db.flag_message_as_suspicious(...)
    return "Sorry I cannot assist with that."
```

### 5. RAG Tool retrieves documents
```python
# Generate embedding
embedding = embedder.encode("How do I withdraw money?")

# Search Supabase
results = db.search_documents(
    embedding=embedding,
    limit=3,
    threshold=0.7
)

# Filter by document source
docs = [r for r in results 
        if r['source'] == 'sgbank_withdrawal_policy_and_procedures']
```

### 6. LLM generates response
```
System prompt: "Use only the withdrawal policy document..."
Documents: [retrieved policy chunks]
Query: "How do I withdraw money?"
Response: "You can withdraw through ATMs, branches, or mobile banking..."
```

### 7. Sentinel validates output
```python
if self._check_sentinel_output(agent_key, user_message, answer):
    # Block and flag
    return "Sorry I cannot assist with that."
```

### 8. Chatbot stores response
```python
db.add_message(
    conversation_id=self.conversation_id,
    user_id=self.user_id,
    role="assistant",
    content=answer,
    metadata={"agent": "withdrawal"}
)
```

### 9. Audit log completion
```python
db.create_audit_log(
    user_id=self.user_id,
    action="response_generated",
    resource="conversation",
    details={"response_length": len(answer)}
)
```

### 10. Return to user
```
Assistant: "You can withdraw through ATMs, branches, or mobile banking..."
```

## User Tracking

### Local Testing (Current)
```python
# Fixed UUID (same user every run)
self.user_id = str(uuid5(NAMESPACE_DNS, "local-test-user"))
# Result: "4b6e3a0f-6f7f-52c8-bd61-6f5c8b5c8b5c"

# Auto-creates on first run
if not db.get_user(self.user_id):
    db.create_user(
        user_id=self.user_id,
        email="local@test.local",
        metadata={"type": "local_chatbot"}
    )
```

### Future With Auth
```python
# Extract from JWT token
user_id = jwt.decode(token)['sub']

# RLS policies enforce per-user data isolation
# Each user only sees their own conversations
```

## Vector Search (RAG)

### Embedding Model
- **Name:** all-MiniLM-L6-v2
- **Dimensions:** 384
- **Speed:** ~2K sentences/second

### Search Algorithm
1. Generate query embedding (384-dim vector)
2. Calculate cosine similarity with all documents
3. Return top-K results with similarity > threshold
4. Filter by document source
5. Pass to LLM as context

### Example
```
Query: "What's the withdrawal limit?"
Embedding: [0.123, -0.456, 0.789, ...]
Similarity scores: [0.87, 0.84, 0.79, 0.71, ...]
Threshold: 0.7
Results: Top 3 documents with similarity > 0.7
```

## Audit Logging

All actions tracked in `audit_logs` table:

| Action | When | Data |
|--------|------|------|
| `message_received` | User sends message | agent routing |
| `response_generated` | Bot responds | response length |
| `chat_error` | Error occurs | error details |

**Retrieve logs:**
```python
logs = db.get_user_audit_logs("local-test-user")
for log in logs:
    print(f"{log['created_at']}: {log['action']}")
```

## Setup & Run

### 1. Create Supabase project
- Enable pgvector extension

### 2. Run schema (in SQL Editor)
[See SQL schema above]

### 3. Set environment variables
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-key
OPENAI_API_KEY=your-key
```

### 4. Ingest documents
```bash
python ingest.py
```

### 5. Run chatbot
```bash
python main.py
```

## Troubleshooting

### "Could not find table 'users'"
- SQL schema not run in Supabase
- Run the schema SQL in Supabase > SQL Editor

### "No relevant excerpts found"
- Documents not ingested
- Run `python ingest.py`
- Check debug output for document count

### "Connection failed"
- Check `.env` has SUPABASE_URL and SUPABASE_SERVICE_KEY
- Verify Supabase project is active

### Empty search results from vector search
- Threshold too high (0.7)
- Too few documents ingested
- Query very different from documents