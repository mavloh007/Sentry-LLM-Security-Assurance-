import os
import uuid
from typing import List, Dict, Tuple
from pypdf import PdfReader
from src.vector_store.vector_store import VectorStore

PDF_PATH = "src/documents"

'''
doc_id's for ref,
doc_id='sgbank_emergency_withdrawal_policy'
doc_id='sgbank_identity_verification_and_authentication_policy'
doc_id='sgbank_transaction_monitoring_and_fraud_detection_policy'
doc_id='sgbank_withdrawal_policy_and_procedures'
'''

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
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
    # stable-ish ids: doc_id + index
    return [f"{doc_id}::chunk_{i:05d}" for i in range(num_chunks)]

def main():
    pdf_files = get_pdf_files(PDF_PATH)
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {PDF_PATH}")

    print(f"Found {len(pdf_files)} PDFs")
    print("Initializing vector store...")
    vs = VectorStore(persist_directory="vectordb")

    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path)
        doc_id = slugify_filename(pdf_path)  # identifiable doc_id from filename

        print(f"\nLoading: {pdf_path}")
        text = load_pdf(pdf_path)

        print("Splitting text...")
        chunks = split_text(text)

        # Attach metadata into the chunk text as a fallback, even if the store
        # doesn't support metadata filters yet.
        chunks = [f"[DOC_ID: {doc_id}]\n[SOURCE: {source_name}]\n{c}" for c in chunks]

        ids = build_ids(doc_id, len(chunks))

        print(f"Adding {len(chunks)} chunks with doc_id='{doc_id}'...")
        # Preferred: pass ids + metadatas if your VectorStore supports it.
        # If it doesn't, this will fall back to plain add_documents below.
        try:
            metadatas: List[Dict] = [
                {"doc_id": doc_id, "source": source_name, "chunk_index": i}
                for i in range(len(chunks))
            ]
            vs.add_documents(chunks, ids=ids, metadatas=metadatas)
        except TypeError:
            # VectorStore.add_documents doesn't accept ids/metadatas yet
            vs.add_documents(chunks)

    # vs.persist()
    print("Ingestion complete.")

if __name__ == "__main__":
    main()