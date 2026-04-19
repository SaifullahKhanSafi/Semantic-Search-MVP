"""
ingestion_pipeline.py
---------------------
Shared ingestion logic used by both api.py (upload endpoint) and ingest.py (CLI).
This ensures both paths use identical chunking, embedding, and deduplication.
"""

import os
import shutil
import pandas as pd
import chromadb

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Shared constants ────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
#DB_DIR = "./chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ".", " ", ""]


def get_embeddings():
    """Return the shared embedding model (cpu by default)."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},  # change to 'cuda' if you have a GPU
    )


def get_db(embeddings=None):
    """Return a connected ChromaDB instance."""
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def load_file(file_path: str, original_filename: str) -> list[Document]:
    """
    Load a single file into LangChain Documents.
    Raises ValueError for unsupported extensions.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()

    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif ext == ".csv":
        try:
            loader = CSVLoader(file_path, encoding="utf-8")
            return loader.load()
        except Exception as e:
            print(f"  CSVLoader failed, falling back to Pandas… ({e})")
            df = pd.read_csv(file_path, encoding_errors="replace", on_bad_lines="skip")
            df = df.head(1000)
            text_content = df.to_string(index=False)
            return [Document(page_content=text_content, metadata={"source": original_filename})]

    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
        text_content = df.to_string(index=False)
        return [Document(page_content=text_content, metadata={"source": original_filename})]

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_existing_sources(db: Chroma) -> set[str]:
    """Return the set of source filenames already stored in ChromaDB."""
    try:
        existing = db.get()
        return {
            os.path.basename(m["source"])
            for m in existing.get("metadatas", [])
            if m and "source" in m
        }
    except Exception:
        return set()


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    return splitter.split_documents(documents)


def ingest_files(
    file_paths: list[tuple[str, str]],  # list of (temp_path, original_filename)
    db: Chroma,
) -> dict:
    """
    Full ingest pipeline:
      1. Skip duplicates
      2. Load each file
      3. Chunk
      4. Embed & store in ChromaDB

    Returns a result dict with keys: ingested, skipped, chunks, errors
    """
    existing_sources = get_existing_sources(db)

    all_documents: list[Document] = []
    skipped: list[str] = []
    errors: list[str] = []

    for temp_path, original_filename in file_paths:
        basename = os.path.basename(original_filename)

        # ── Duplicate check ──────────────────────────────────────────────
        if basename in existing_sources:
            print(f"  [SKIP] {basename} already in database.")
            skipped.append(basename)
            continue

        # ── Load ─────────────────────────────────────────────────────────
        try:
            print(f"  [LOAD] {basename}")
            docs = load_file(temp_path, original_filename)

            # Normalise the source metadata to just the filename (not temp path)
            for doc in docs:
                doc.metadata["source"] = basename

            all_documents.extend(docs)

        except ValueError as e:
            print(f"  [ERROR] {basename}: {e}")
            errors.append(f"{basename}: {e}")
        except Exception as e:
            print(f"  [ERROR] {basename}: {e}")
            errors.append(f"{basename}: {e}")

    # ── Nothing new to ingest ────────────────────────────────────────────
    if not all_documents:
        return {
            "ingested": 0,
            "skipped": skipped,
            "chunks": 0,
            "errors": errors,
        }

    # ── Chunk ────────────────────────────────────────────────────────────
    print(f"  [CHUNK] Splitting {len(all_documents)} document(s)…")
    chunks = chunk_documents(all_documents)
    print(f"  [CHUNK] Created {len(chunks)} chunks.")

    # ── Embed & Store ────────────────────────────────────────────────────
    print(f"  [EMBED] Storing in ChromaDB…")
    db.add_documents(chunks)
    print(f"  [DONE]  {len(chunks)} chunks stored.")

    ingested_count = len(all_documents)  # original doc count, not chunks

    return {
        "ingested": ingested_count,
        "skipped": skipped,
        "chunks": len(chunks),
        "errors": errors,
    }

def get_db(embeddings=None):
    """Return a connected ChromaDB instance hosted in Docker."""
    if embeddings is None:
        embeddings = get_embeddings()
        
    # Connect to the Chroma Docker container using the service name 'chroma-server'
    chroma_client = chromadb.HttpClient(host='chroma-server', port=8000)
    
    return Chroma(
        client=chroma_client, 
        embedding_function=embeddings)