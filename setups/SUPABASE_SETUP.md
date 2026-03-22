# Supabase Setup Guide for Withdrawal Chatbot

## Quick Start (5 Steps)

1. Create Supabase project
2. Enable pgvector extension
3. Run database schema SQL
4. Add credentials to `.env` file
5. Run `python ingest.py` to populate documents

Then start chatting with `python main.py`!

## System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      YOUR LOCAL COMPUTER                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Python Application                     │  │
│  │                                                          │  │
│  │  main.py ──────────► WithdrawalChatbot ◄──── ingest.py  │  │
│  │   (CLI)              (Chat Logic)           (PDF Upload) │  │
│  │                          │                                │  │
│  │                          ▼                                │  │
│  │              SupabaseDB (API Wrapper)                    │  │
│  │         (src/db/supabase_client.py)                      │  │
│  │                          │                                │  │
│  │                    Uses .env credentials                 │  │
│  │            (SUPABASE_URL, SUPABASE_SERVICE_KEY)          │  │
│  └──────────────────────────┬──────────────────────────────┘  │
│                             │                                  │
└─────────────────────────────┼──────────────────────────────────┘
                              │
                    HTTPS Connection (Secure)
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   SUPABASE CLOUD (PostgreSQL)                  │
│                                                                  │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐    │
│  │    messages    │  │  conversations  │  │    users     │    │
│  │   (chat hist)  │  │  (sessions)     │  │  (profiles)  │    │
│  └────────────────┘  └─────────────────┘  └──────────────┘    │
│           │                   │                    │            │
│           └───────────────────┼────────────────────┘            │
│                               │                                 │
│  ┌────────────────────────────┼────────────────────────┐       │
│  │           pgvector (Vector Search)                  │       │
│  │                                                    │       │
│  │  ┌──────────────────────────────────────────────┐  │       │
│  │  │   documents (with 384-dim embeddings)       │  │       │
│  │  │                                              │  │       │
│  │  │  • sgbank_withdrawal_policy: 18 docs        │  │       │
│  │  │  • sgbank_emergency_policy: 8 docs          │  │       │
│  │  │  • sgbank_identity_policy: 2 docs           │  │       │
│  │  │  • sgbank_fraud_policy: 54 docs             │  │       │
│  │  │                                              │  │       │
│  │  └──────────────────────────────────────────────┘  │       │
│  └────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  audit_logs (Tracking)  │  message_flags (Security)    │   │
│  │  • All actions logged   │  • Suspicious content        │   │
│  │  • User ID tracked      │  • Reason + details          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow Example

```
Step 1: User types question
  You: "How do I withdraw?"
                │
                ▼
Step 2: Store in Supabase
  → messages table (role='user')
  → conversations table (grouping)
                │
                ▼
Step 3: Generate embedding & search
  → Encode: "How do I withdraw?" → [384 numbers]
  → Search pgvector for similar documents
  → Return top 3 matching policy chunks
                │
                ▼
Step 4: Send to LLM with context
  System: "Use only withdrawal policy..."
  Context: [3 relevant policy documents]
  Query: "How do I withdraw?"
                │
                ▼
Step 5: Validate & store response
  → Check with Sentinel Guard
  → Save assistant response to messages table
  → Create audit log entry
                │
                ▼
Step 6: Return to user
  Assistant: "You can withdraw through ATMs, 
             branches, or mobile banking..."
```

## Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Sign up or log in
3. Click "New Project"
4. Fill in project details:
   - **Name:** `sgbank-withdrawal-chatbot`
   - **Database Password:** Create a strong password (save this!)
   - **Region:** Choose closest to your location
5. Wait for database to initialize (this takes 5-10 minutes)

## Step 2: Get Your API Keys

1. In your Supabase project, go to **Settings** → **API**
2. You'll see these keys (you need to copy both):
   - **Project URL** - This is your `SUPABASE_URL`
   - **Service Role Key** - This is your `SUPABASE_SERVICE_KEY` (use this for backend)

**⚠️ IMPORTANT:** Keep Service Role Key SECRET! Never share or commit to git.

## Step 3: Enable pgvector Extension

1. In your Supabase project, go to **SQL Editor**
2. Click **"New Query"**
3. Copy and paste this SQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

4. Click **"Run"**

You should see "Query successful" ✓

## Step 4: Create Database Tables

1. In **SQL Editor**, click **"New Query"** again
2. Copy and paste this entire SQL schema:

```sql
-- Create users table
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create conversations table
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create messages table
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create documents table (stores policy documents with embeddings)
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT NOT NULL,
  embedding vector(384),
  source TEXT,
  doc_type TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster vector search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create audit_logs table (tracks all actions)
CREATE TABLE audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  action TEXT NOT NULL,
  resource_type TEXT,
  details JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create message_flags table (for suspicious content)
CREATE TABLE message_flags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  reason TEXT NOT NULL,
  details JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create sessions table
CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP
);
```

3. Click **"Run"** 

You should see "Query successful" ✓ for each table created

## Step 5: Add Credentials to .env File

1. Open `.env` file in your project root (create it if it doesn't exist)
2. Add these lines with your actual Supabase credentials:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-name.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# OpenAI (required for chatbot responses)
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini

# Optional: Sentinel Guard for safety validation
SENTINEL_API_KEY=your-sentinel-key-here
```

**Where to find each value:**

- **SUPABASE_URL:** Supabase Dashboard → Settings → API → Copy "Project URL"
- **SUPABASE_SERVICE_KEY:** Supabase Dashboard → Settings → API → Copy "Service Role Key"
- **OPENAI_API_KEY:** OpenAI Platform → API Keys → Create new secret key

## Step 6: Populate Documents with `ingest.py`

Now that your database is set up, ingest the PDF policy documents:

```bash
python ingest.py
```

This script will:
- Find all PDF files in `src/documents/`
- Extract text from each PDF
- Split into 800-character chunks (100 char overlap)
- Generate 384-dimensional embeddings
- Store in Supabase with metadata

Expected output:
```
🚀 STARTING PDF INGESTION TO SUPABASE VECTOR STORE

📄 Found 4 PDFs

🔌 Connecting to Supabase...
✓ Connected to Supabase

📦 Loading embedding model (all-MiniLM-L6-v2)...
✓ Embedding model ready

📥 Processing: sgbank_withdrawal_policy_and_procedures.pdf
   ✓ Loaded 12345 characters
   ✓ Split into 18 chunks
   ✓ Ingested 18/18 chunks

[... more PDFs ...]

✅ Ingestion complete: 82 total documents
```

## Step 7: Test the Connection

Run the chatbot to test everything is working:

```bash
python main.py
```

Expected output:
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

You: 
```

If you see this, you're all set! ✓

Try asking: `How do I withdraw?`

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point - runs the chatbot |
| `ingest.py` | Ingests PDFs into Supabase vector database |
| `src/chatbot/withdrawal_chatbot.py` | Main chatbot with 4 policy agents |
| `src/db/supabase_client.py` | Database API wrapper (482 lines) |
| `.env` | Your environment configuration (credentials) |

## Database Schema Overview

```
users
├─ id (UUID)
├─ email
└─ metadata

conversations
├─ id (UUID)
├─ user_id → users
└─ title

messages
├─ id (UUID)
├─ conversation_id → conversations
├─ user_id → users
├─ role ('user' or 'assistant')
└─ content

documents (Vector Store)
├─ id (UUID)
├─ content
├─ embedding (vector 384-dim)
├─ source (e.g., 'sgbank_withdrawal_policy_and_procedures')
├─ doc_type
└─ metadata

audit_logs (Audit Trail)
├─ id (UUID)
├─ user_id → users
├─ action (e.g., 'message_received')
├─ resource_type
└─ details (JSONB)

message_flags (Suspicious Content)
├─ id (UUID)
├─ message_id → messages
├─ reason
└─ details

sessions
├─ id (UUID)
├─ user_id → users
└─ expires_at
```

## How the Chatbot Works

```
1. User: "How do I withdraw?"
   ↓
2. main.py receives input
   ↓
3. WithdrawalChatbot.chat()
   ├─ Store message in Supabase
   ├─ Route to "withdrawal" agent
   ├─ Generate embedding of query
   └─ Search Supabase pgvector for similar documents
   ↓
4. RAG Retrieval
   ├─ Find top 3 matching policy documents
   └─ Pass to LLM with system prompt
   ↓
5. LLM Response Generation (GPT-4o-mini)
   ├─ Validate with Sentinel Guard
   ├─ Generate answer based on documents
   └─ Return response
   ↓
6. Store response in Supabase
   ├─ Save assistant message
   ├─ Create audit log
   └─ Return to user
```

## Troubleshooting

### Error: "Could not find the table 'public.users'"
**Solution:** SQL schema wasn't run. Go back to Step 4 and run the SQL script in Supabase SQL Editor.

### Error: "SUPABASE_URL or SUPABASE_SERVICE_KEY not found"
**Solution:** Check your `.env` file:
- Make sure the file exists in project root
- Make sure it has `SUPABASE_URL=` and `SUPABASE_SERVICE_KEY=`
- Restart Python after editing `.env`

### Error: "No relevant excerpts found"
**Solution:** Documents not ingested yet. Run `python ingest.py` first (see Step 6).

### Vector search returns 0 results
**Solution:** 
- Make sure `ingest.py` completed successfully
- Check that documents table has data: `SELECT COUNT(*) FROM documents;` in SQL Editor
- Lower the similarity threshold (currently 0.7)

### Embedding dimension mismatch
**Solution:** All embeddings must be 384-dim (for all-MiniLM-L6-v2 model).
- Check ingest.py output for errors
- Verify `src/documents/` has the policy PDFs

## Security Notes

1. **Service Role Key** - Keep secret! Only use in backend (Python)
2. **.env file** - Never commit to git. Add to `.gitignore`
3. **Row Level Security** - Database enforces per-user data isolation
4. **Audit Logs** - All actions tracked for security review
5. **Message Flags** - Suspicious content automatically flagged

## Next Steps

After setup is complete, you can:
- Run `python main.py` to chat with the bot
- Query `audit_logs` to see all actions
- Query `message_flags` to see flagged messages
- Query `messages` to see conversation history
