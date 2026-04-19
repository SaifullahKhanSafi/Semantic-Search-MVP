"""
ingest.py  —  CLI batch ingestion
----------------------------------
Drop files into the /data folder and run this script to build the database.
Uses the exact same pipeline as the API upload endpoint.
"""

import os
from ingestionpipeline import get_embeddings, get_db, ingest_files

DATA_DIR = "data"


def build_database():
    print(f"Scanning '{DATA_DIR}' folder…")

    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' folder not found. Please create it and add files.")
        return

    # Collect all files from the data directory
    file_pairs: list[tuple[str, str]] = []
    for filename in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(full_path):
            file_pairs.append((full_path, filename))

    if not file_pairs:
        print("No files found in the data folder. Exiting.")
        return

    print(f"Found {len(file_pairs)} file(s). Loading embedding model…")
    embeddings = get_embeddings()
    db = get_db(embeddings)

    # Run through the shared ingestion pipeline
    result = ingest_files(file_pairs, db)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Ingestion Summary ──────────────────────────────────────")
    print(f"  Files ingested : {result['ingested']}")
    print(f"  Chunks stored  : {result['chunks']}")
    print(f"  Duplicates     : {len(result['skipped'])} → {result['skipped'] or 'none'}")
    print(f"  Errors         : {len(result['errors'])} → {result['errors'] or 'none'}")
    print("───────────────────────────────────────────────────────────")


if __name__ == "__main__":
    build_database()