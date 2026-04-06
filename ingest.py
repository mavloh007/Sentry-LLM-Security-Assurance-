import os
import sys
from typing import List, Dict, Tuple
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db.supabase_client import SupabaseDB, SupabaseVectorStore

load_dotenv()

PDF_PATH = "src/documents"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
TEST_SEARCH_THRESHOLD = float(os.getenv("INGEST_TEST_SEARCH_THRESHOLD", "0.5"))

'''
doc_id's for ref,
doc_id='sgbank_emergency_withdrawal_policy'
doc_id='sgbank_identity_verification_and_authentication_policy'
doc_id='sgbank_transaction_monitoring_and_fraud_detection_policy'
doc_id='sgbank_withdrawal_policy_and_procedures'
'''

def load_pdf(path: str) -> str:
    """Load text from PDF file"""
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks


def get_pdf_files(folder: str) -> List[str]:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]

def slugify_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0].strip().lower()
    # simple slug: keep alnum, replace others with underscore
    out = []
    prev_us = False
    for ch in name:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    slug = "".join(out).strip("_")
    return slug or "document"

def build_ids(doc_id: str, num_chunks: int) -> List[str]:
    """Generate stable IDs for chunks (kept for reference)"""
    return [f"{doc_id}::chunk_{i:05d}" for i in range(num_chunks)]


def main():
    """Main ingestion pipeline: PDF → Chunks → Embeddings → Supabase"""
    
    print("\n" + "="*70)
    print("🚀 STARTING PDF INGESTION TO SUPABASE VECTOR STORE")
    print("="*70 + "\n")
    
    # 1. Find all PDFs
    pdf_files = get_pdf_files(PDF_PATH)
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {PDF_PATH}")

    print(f"📄 Found {len(pdf_files)} PDFs\n")

    # 2. Initialize Supabase DB and embedder
    print("🔌 Connecting to Supabase...")
    try:
        db = SupabaseDB()
        if not db.health_check():
            print("❌ Failed to connect to Supabase")
            return False
        print("✓ Connected to Supabase\n")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

    print(f"📦 Initializing OpenAI embeddings ({EMBEDDING_MODEL}, {EMBEDDING_DIMENSIONS} dims)...")
    openai_client = OpenAI()
    vs = SupabaseVectorStore(db)
    print("✓ Embedding client ready\n")

    # 3. Process each PDF
    total_chunks_ingested = 0
    total_failed = 0

    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path)
        doc_id = slugify_filename(pdf_path)

        print(f"📥 Processing: {source_name} (doc_id: {doc_id})")
        
        try:
            # Load PDF
            text = load_pdf(pdf_path)
            print(f"   ✓ Loaded {len(text)} characters")

            # Split into chunks
            chunks = split_text(text, chunk_size=800, overlap=100)
            print(f"   ✓ Split into {len(chunks)} chunks")

            # Generate embeddings and ingest
            batch_results = []
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    resp = openai_client.embeddings.create(
                        input=chunk,
                        model=EMBEDDING_MODEL,
                        dimensions=EMBEDDING_DIMENSIONS,
                    )
                    embedding = resp.data[0].embedding
                    
                    # Insert into Supabase
                    result = db.add_document(
                        content=chunk,
                        embedding=embedding,
                        source=doc_id,
                        doc_type="policy",
                        metadata={
                            "source_filename": source_name,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "original_doc_id": doc_id
                        }
                    )
                    batch_results.append(result)
                    
                    # Progress indicator
                    if (i + 1) % 5 == 0:
                        print(f"      → Ingested {i + 1}/{len(chunks)} chunks")
                        
                except Exception as e:
                    print(f"      ⚠️  Error ingesting chunk {i}: {e}")
                    total_failed += 1
                    continue

            total_chunks_ingested += len(batch_results)
            print(f"   ✓ Successfully ingested {len(batch_results)} chunks\n")

        except Exception as e:
            print(f"   ❌ Error processing {source_name}: {e}\n")
            total_failed += 1
            continue

    # 4. Verification
    print("\n" + "="*70)
    print("✅ INGESTION COMPLETE!")
    print("="*70)
    print(f"Total chunks ingested: {total_chunks_ingested}")
    print(f"Errors encountered: {total_failed}\n")

    # 5. Test vector search
    if total_chunks_ingested > 0:
        print("🧪 Testing vector search...\n")
        test_queries = [
            "withdrawal policy",
            "emergency withdrawal",
            "identity verification"
        ]
        
        for query in test_queries:
            resp = openai_client.embeddings.create(
                input=query,
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            query_embedding = resp.data[0].embedding
            results = vs.search(query_embedding, limit=2, threshold=TEST_SEARCH_THRESHOLD)
            
            if results:
                print(f"Query: '{query}'")
                for j, result in enumerate(results, 1):
                    similarity = result.get('similarity', 0)
                    content = result.get('content', '')[:70]
                    source = result.get('source', 'unknown')
                    print(f"  {j}. [{similarity:.2%}] {content}... ({source})")
                print()
            else:
                print(f"Query: '{query}' - No results found\n")

    return total_chunks_ingested > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)