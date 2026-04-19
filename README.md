# Project Khudi: Enterprise RAG & Semantic Search Engine

An advanced, production-ready **Retrieval-Augmented Generation (RAG)** backend and semantic search system.  
This project implements a **two-stage retrieval pipeline (Bi-Encoder + Cross-Encoder)** to ensure highly accurate and context-aware document retrieval.

---

## Architecture & Tech Stack

| Layer        | Technology |
|-------------|-----------|
| **Backend** | FastAPI (Python) |
| **Vector DB** | ChromaDB (Dockerized HTTP service) |
| **Ingestion** | LangChain (PDF, DOCX, CSV loaders) |
| **Frontend** | React.js (Vite) |
| **Infra** | Docker & Docker Compose (WSL 2 optimized) |

---

## Models & Benchmark Data

###  Dense Retrieval (Bi-Encoder)
- **Model:** `BAAI/bge-base-en-v1.5`
- **Role:** Converts documents & queries into embeddings and retrieves top **N candidates**
- **Performance:** Top-ranked in **MTEB (Massive Text Embedding Benchmark)**

---

###  Re-Ranking (Cross-Encoder)
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Role:** Scores retrieved chunks against the query for precise ranking
- **Dataset:** MS MARCO
- **Advantage:** High accuracy with low latency

---

## Setup & Installation

### Prerequisites
- Docker Desktop (**WSL 2 enabled on Windows**)
- Node.js & npm
- 5–10 GB free disk space

---

### Step 1: Start Backend

```bash
docker compose up --build
```

**Expected Output:**
```
INFO: Application startup complete.
```

---

### Step 2: Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Access UI at:
```
http://localhost:5173
```

---

##  Usage Guide

### Upload Documents
- Upload PDFs, DOCX, CSV via UI
- Backend:
  - Stores in `temp_uploads/`
  - Splits text (chunk size: 1000, overlap: 200)
  - Generates embeddings
  - Stores in ChromaDB

---

### Search
- Enter natural language query
- Pipeline:
  1. Retrieve top 10 chunks (Bi-Encoder)
  2. Re-rank (Cross-Encoder)
  3. Return best matches

---

###  CLI Batch Ingestion

```bash
docker compose exec api python ingest.py
```

---

## Technical Optimizations

- **Networked ChromaDB**
  - Runs as independent microservice
  - Persistent storage via volume (`./chroma_db_data`)

- **Model Caching**
  - HuggingFace cache mapped to local system
  - Avoids repeated downloads

- **Local-First AI**
  - No external APIs (OpenAI/Anthropic)
  - Fully offline processing

---

##  Project Structure


```
semantic_search_mvp
├─ .dockerignore
├─ api.py
├─ benchmark_bierSciFact.py
├─ benchmark_msmarco.py
├─ benchmark_results.csv
├─ bier_benchmark_eval.py
├─ chroma_db
│  ├─ chroma.sqlite3
│  └─ f23ad2ed-9526-4cd0-a69e-2a29d7d02fa8
│     └─ index_metadata.pickle
├─ cleardb.py
├─ datasets
│  └─ msmarco.zip
├─ docker-compose.yml
├─ Dockerfile
├─ frontend
│  ├─ eslint.config.js
│  ├─ index.html
│  ├─ package-lock.json
│  ├─ package.json
│  ├─ public
│  │  ├─ favicon.svg
│  │  └─ icons.svg
│  ├─ README.md
│  ├─ src
│  │  ├─ App.css
│  │  ├─ App.jsx
│  │  ├─ assets
│  │  │  ├─ hero.png
│  │  │  ├─ react.svg
│  │  │  └─ vite.svg
│  │  ├─ index.css
│  │  └─ main.jsx
│  └─ vite.config.js
├─ ingest.py
├─ ingestionpipeline.py
├─ latency_distribution.png
├─ mrr_comparison.png
├─ ndcg_comparison.png
├─ precision_recall.png
├─ requirements.txt
└─ search.py

```

---

## Author

**Saifullah Khan (22i-1334)**  
Project Khudi – Enterprise Semantic Search Engine

---


