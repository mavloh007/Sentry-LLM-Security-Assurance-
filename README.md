# Joint-Intern-Project

A chatbot framework with vector database storage and prompt attack testing capabilities.

## Features

- **Chatbot**: Conversational AI with RAG (Retrieval-Augmented Generation) capabilities
- **Vector Store**: Efficient document storage and similarity search using Supabase pgvector (384-dim embeddings)
- **Prompt Attack Dataset**: Security testing dataset with 15+ prompt injection patterns
- **Frontend Skeleton**: A basic UI for interacting with the chatbot (placeholder mode).

## Getting Started

```
.
├── src/
│   ├── chatbot/           # Chatbot implementation with RAG
│   │   ├── __init__.py
│   │   └── chatbot.py
│   ├── vector_store/      # Vector database for embeddings
│   │   ├── __init__.py
│   │   └── vector_store.py
│   └── datasets/          # Prompt attack dataset
│       ├── __init__.py
│       └── prompt_attacks.py
├── tests/                 # Unit tests
├── examples.py           # Example usage
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/isaroyston/Joint-Intern-Project.git
cd Joint-Intern-Project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Quick Start

### Vector Store Usage

```python
from src.vector_store import VectorStore

# Initialize vector store
vector_store = VectorStore(
    persist_directory="./data/vector_db",
    collection_name="my_collection"
)

# Add documents
documents = [
    "Python is a programming language.",
    "Machine learning is a subset of AI."
]
vector_store.add_documents(documents)

# Search for similar documents
results = vector_store.search("What is Python?", n_results=2)
print(results['documents'])
```

### Chatbot Usage

```python
from src.chatbot import Chatbot
from src.vector_store import VectorStore

# Initialize with vector store for RAG
vector_store = VectorStore()
vector_store.add_documents(["Your knowledge base documents here"])

# Create chatbot
chatbot = Chatbot(
    api_key="your-openai-api-key",
    vector_store=vector_store
)

# Chat with RAG
response = chatbot.chat("Your question here", use_rag=True)
print(response)
```

### Prompt Attack Testing

```python
from src.datasets import PromptAttackDataset

# Load attack dataset
dataset = PromptAttackDataset()

# Get all attacks
attacks = dataset.get_all_attacks()
print(f"Total attacks: {len(attacks)}")

# Filter by severity
high_severity = dataset.get_attacks_by_severity("high")

# Filter by category
injection_attacks = dataset.get_attacks_by_category("instruction_override")

# Test chatbot security
results = dataset.test_chatbot(chatbot, log_file="results.json")
```

## Running Examples

```bash
python examples.py
```

## Components

### 1. Vector Store (`src/vector_store/`)
- Uses ChromaDB for efficient vector storage
- Sentence transformers for embeddings
- Supports document addition, search, and management

### 2. Chatbot (`src/chatbot/`)
- OpenAI GPT integration
- RAG capabilities with vector store
- Conversation history management
- Customizable system prompts

### 3. Prompt Attack Dataset (`src/datasets/`)
- 15+ predefined attack patterns
- Categories: injection, jailbreak, social engineering, etc.
- Severity levels: high, medium, low
- Testing capabilities for chatbot security

## Attack Categories

- **instruction_override**: Direct attempts to override system instructions
- **role_manipulation**: Trying to change the AI's role or persona
- **information_extraction**: Attempting to extract system prompts or config
- **jailbreak**: Bypassing safety measures
- **social_engineering**: Using manipulation tactics
- **encoding_attack**: Using encoding to hide malicious intent
- **context_manipulation**: Injecting fake context
- **technical_exploit**: Using special tokens or technical tricks

## Environment Variables

```env
OPENAI_API_KEY=your_api_key_here
SENTINEL_API_KEY=your_sentinel_api_key_here
SENTINEL_API_URL=https://sentinel.stg.aiguardian.gov.sg/api/v1/validate
VECTOR_DB_PATH=./data/vector_db
COLLECTION_NAME=chatbot_knowledge
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=500
```

## Development

### Running Tests
```bash
# Tests can be added in the tests/ directory
pytest tests/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.