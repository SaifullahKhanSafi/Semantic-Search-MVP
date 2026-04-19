from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# ==========================================
# 1. DOWNLOAD THE PUBLIC DATASET (SciFact)
# ==========================================
print("Downloading BEIR/SciFact Corpus (5,183 documents)...")
corpus_dataset = load_dataset("BeIR/scifact", "corpus", split="corpus")

print("Downloading BEIR/SciFact Queries & Answer Keys...")
queries_dataset = load_dataset("BeIR/scifact", "queries", split="queries")
qrels_dataset = load_dataset("BeIR/scifact-qrels", split="test") # The Ground Truth

# ==========================================
# 2. PREPARE THE DATA
# ==========================================
print("Preparing Corpus for Ingestion...")
documents = []
# Map corpus IDs to their text so we can easily check the answers later
corpus_mapping = {} 

for row in corpus_dataset:
    doc_id = row["_id"]
    # Combine title and text for better context
    text = f"{row['title']}. {row['text']}" 
    documents.append(Document(page_content=text, metadata={"doc_id": doc_id}))
    corpus_mapping[doc_id] = text

print("Mapping Answer Keys to Queries...")
# Convert queries dataset to a dictionary for fast lookup
query_dict = {q["_id"]: q["text"] for q in queries_dataset}

evaluation_data = []
# qrels contains: query-id, corpus-id, and a score (usually 1 for a match)
for row in qrels_dataset:
    q_id = str(row["query-id"])
    doc_id = str(row["corpus-id"])
    
    if q_id in query_dict and doc_id in corpus_mapping:
        evaluation_data.append({
            "query": query_dict[q_id],
            "correct_doc_id": doc_id
        })

# ==========================================
# 3. BUILD THE ENGINE
# ==========================================
print(f"Loading BGE-Base Embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'} # Change to 'cuda' if you have an Nvidia GPU
)

print(f"Building Ephemeral Chroma DB (This will take a few minutes)...")
# We use an in-memory DB so it doesn't overwrite your project files
db = Chroma.from_documents(documents=documents, embedding=embeddings)

print(f"Loading Cross-Encoder Re-ranker...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==========================================
# 4. RUN THE BENCHMARK
# ==========================================
print(f"\nRunning SciFact Benchmark ({len(evaluation_data)} queries)...\n")
hits = 0
mrr_score = 0.0
K_VALUE = 3

for i, test in enumerate(evaluation_data):
    query = test["query"]
    expected_doc_id = test["correct_doc_id"]
    
    # Stage 1: Vector Search
    initial_results = db.similarity_search(query, k=10)
    
    # Stage 2: Re-Rank
    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = cross_encoder.predict(pairs)
    
    scored_results = zip(scores, initial_results)
    sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
    top_k_results = sorted_results[:K_VALUE]
    
    found_match = False
    for rank, (score, result) in enumerate(top_k_results):
        # Check if the metadata ID matches the expected answer ID
        if result.metadata["doc_id"] == expected_doc_id:
            found_match = True
            hits += 1
            mrr_score += 1.0 / (rank + 1)
            break 

# ==========================================
# 5. FINAL SCORES
# ==========================================
if len(evaluation_data) > 0:
    hit_rate = (hits / len(evaluation_data)) * 100
    mrr = mrr_score / len(evaluation_data)
    
    print("="*50)
    print("PUBLIC BEIR: SCIFACT BENCHMARK RESULTS")
    print("="*50)
    print(f"Corpus Size: {len(documents)} Documents")
    print(f"Queries Tested: {len(evaluation_data)}")
    print(f"Hit Rate (@{K_VALUE}): {hit_rate:.1f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print("="*50)