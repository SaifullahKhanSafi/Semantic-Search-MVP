import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# ---------------- CONFIG ----------------
K = 5
TEST_SIZE = 100
CSV_FILE = "benchmark_results.csv"

# ---------------- METRICS ----------------
def precision_at_k(relevant, retrieved, k):
    return len(set(retrieved[:k]) & set(relevant)) / k

def recall_at_k(relevant, retrieved, k):
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)

def dcg_at_k(scores, k):
    scores = np.array(scores[:k])
    return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

def ndcg_at_k(scores, k):
    ideal = sorted(scores, reverse=True)
    return dcg_at_k(scores, k) / (dcg_at_k(ideal, k) + 1e-8)

# ---------------- LOAD DATA ----------------
print("Loading MS MARCO dataset...")
dataset = load_dataset("ms_marco", "v1.1", split="validation")

documents = []
evaluation_data = []

for i in range(TEST_SIZE):
    row = dataset[i]
    query = row["query"]
    passages = row["passages"]["passage_text"]
    labels = row["passages"]["is_selected"]

    relevant_docs = []

    for idx, text in enumerate(passages):
        documents.append(Document(page_content=text))
        if labels[idx] == 1:
            relevant_docs.append(text)

    if relevant_docs:
        evaluation_data.append({
            "query": query,
            "relevant_docs": relevant_docs
        })

print(f"Total queries: {len(evaluation_data)}")
print(f"Total documents: {len(documents)}")

# ---------------- EMBEDDING + DB ----------------
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

print("Building vector DB...")
db = Chroma.from_documents(documents, embeddings)

# ---------------- RE-RANKER ----------------
print("Loading cross-encoder...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ---------------- STORAGE ----------------
results = []

# ---------------- EVALUATION LOOP ----------------
print("Running evaluation...\n")

for idx, test in enumerate(evaluation_data):
    query = test["query"]
    relevant = test["relevant_docs"]

    # -------- LATENCY --------
    start = time.time()
    initial_results = db.similarity_search(query, k=10)
    latency = time.time() - start

    retrieved = [doc.page_content for doc in initial_results]

    # -------- BEFORE RE-RANK --------
    mrr = 0
    scores = []

    for rank, doc in enumerate(retrieved):
        if doc in relevant:
            if mrr == 0:
                mrr = 1 / (rank + 1)
            scores.append(1)
        else:
            scores.append(0)

    precision = precision_at_k(relevant, retrieved, K)
    recall = recall_at_k(relevant, retrieved, K)
    ndcg = ndcg_at_k(scores, K)

    # -------- RE-RANK --------
    pairs = [[query, doc] for doc in retrieved]
    ce_scores = cross_encoder.predict(pairs)

    reranked = [doc for _, doc in sorted(zip(ce_scores, retrieved), reverse=True)]

    # -------- AFTER RE-RANK --------
    mrr_rerank = 0
    scores_rerank = []

    for rank, doc in enumerate(reranked):
        if doc in relevant:
            if mrr_rerank == 0:
                mrr_rerank = 1 / (rank + 1)
            scores_rerank.append(1)
        else:
            scores_rerank.append(0)

    ndcg_rerank = ndcg_at_k(scores_rerank, K)

    # -------- SAVE --------
    results.append([
        idx,
        latency,
        precision,
        recall,
        ndcg,
        mrr,
        mrr_rerank,
        ndcg_rerank
    ])

# ---------------- SAVE CSV ----------------
print("Saving CSV...")

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "query_id",
        "latency",
        "precision@k",
        "recall@k",
        "ndcg@k",
        "mrr_before",
        "mrr_after",
        "ndcg_after"
    ])
    writer.writerows(results)

print(f"Saved to {CSV_FILE}")

# ---------------- AGGREGATE ----------------
data = np.array(results)

avg_latency = np.mean(data[:,1])
avg_precision = np.mean(data[:,2])
avg_recall = np.mean(data[:,3])
avg_ndcg = np.mean(data[:,4])
avg_mrr_before = np.mean(data[:,5])
avg_mrr_after = np.mean(data[:,6])
avg_ndcg_after = np.mean(data[:,7])

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Latency (ms): {avg_latency*1000:.2f}")
print(f"Precision@{K}: {avg_precision:.3f}")
print(f"Recall@{K}: {avg_recall:.3f}")
print(f"nDCG@{K}: {avg_ndcg:.3f}")
print(f"MRR Before: {avg_mrr_before:.3f}")
print(f"MRR After: {avg_mrr_after:.3f}")
print(f"nDCG After: {avg_ndcg_after:.3f}")
print("="*50)

# ---------------- PLOTS ----------------

# 1. MRR Comparison
plt.figure()
plt.bar(["Before", "After"], [avg_mrr_before, avg_mrr_after])
plt.title("MRR Comparison")
plt.ylabel("MRR")
plt.savefig("mrr_comparison.png")
plt.close()

# 2. nDCG Comparison
plt.figure()
plt.bar(["Before", "After"], [avg_ndcg, avg_ndcg_after])
plt.title("nDCG Comparison")
plt.ylabel("nDCG")
plt.savefig("ndcg_comparison.png")
plt.close()

# 3. Precision vs Recall
plt.figure()
plt.bar(["Precision", "Recall"], [avg_precision, avg_recall])
plt.title("Precision vs Recall")
plt.ylabel("Score")
plt.savefig("precision_recall.png")
plt.close()

# 4. Latency Distribution
plt.figure()
plt.hist(data[:,1])
plt.title("Latency Distribution")
plt.xlabel("Seconds")
plt.ylabel("Frequency")
plt.savefig("latency_distribution.png")
plt.close()

print("Plots saved: mrr_comparison.png, ndcg_comparison.png, precision_recall.png, latency_distribution.png")