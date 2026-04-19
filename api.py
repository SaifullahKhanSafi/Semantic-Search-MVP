import os
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder

# ── Shared pipeline ──────────────────────────────────────────────────────────
from ingestionpipeline import get_embeddings, get_db, ingest_files

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load models once ────────────────────────────────────────────────
print("Loading embedding model…")
embeddings = get_embeddings()
db = get_db(embeddings)

print("Loading Cross-Encoder re-ranker…")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ── Search ───────────────────────────────────────────────────────────────────
@app.get("/search")
def search_database(query: str, k: int = 3):
    initial_results = db.similarity_search(query, k=10)

    if not initial_results:
        return {"query": query, "results": []}

    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = cross_encoder.predict(pairs)
    sorted_results = sorted(zip(scores, initial_results), key=lambda x: x[0], reverse=True)

    formatted_results = []
    for score, result in sorted_results[:k]:
        formatted_results.append({
            "text": result.page_content,
            "source": result.metadata.get("source", "Unknown"),
            "confidence_score": float(score),
        })

    return {"query": query, "results": formatted_results}


# ── Upload ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    1. Save uploaded files to a temp directory
    2. Run them through the full ingest pipeline (chunk -> embed -> store)
    3. Clean up temp files regardless of success or failure
    """
    os.makedirs("temp_uploads", exist_ok=True)

    saved_files: list[tuple[str, str]] = []

    try:
        # ── Save uploads to disk ─────────────────────────────────────────
        for file in files:
            temp_path = os.path.join("temp_uploads", file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append((temp_path, file.filename))

        # ── Run the shared ingestion pipeline ────────────────────────────
        result = ingest_files(saved_files, db)

    finally:
        # ── Always clean up temp folder (even on exception) ──────────────
        if os.path.exists("temp_uploads"):
            shutil.rmtree("temp_uploads")

    # ── Build a human-readable response ──────────────────────────────────
    if result["errors"] and result["ingested"] == 0:
        return {"error": f"All files failed to process: {'; '.join(result['errors'])}"}

    parts = []
    if result["ingested"] > 0:
        parts.append(
            f"Successfully ingested {result['ingested']} file(s) into {result['chunks']} searchable chunks."
        )
    if result["skipped"]:
        parts.append(f"Skipped {len(result['skipped'])} duplicate(s): {', '.join(result['skipped'])}.")
    if result["errors"]:
        parts.append(f"Errors: {'; '.join(result['errors'])}.")

    return {"message": " ".join(parts)}


# ── Clear DB ─────────────────────────────────────────────────────────────────
@app.delete("/clear")
def clear_database():
    try:
        db.delete_collection()
        return {"message": "Database cleared successfully."}
    except Exception as e:
        return {"error": f"Failed to clear database: {str(e)}"}