import os
import csv
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------- CONFIG ----------------
DATASETS = [
    "msmarco",
    "fiqa",
    "nq",
    "trec-covid"
]

K_VALUES = [1, 3, 5, 10]

# ---------------- MODEL ----------------
print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Wrap for BEIR
retriever = DenseRetrievalExactSearch(model, batch_size=32)
evaluator = EvaluateRetrieval(retriever, score_function="cos_sim")

# Optional reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- STORAGE ----------------
summary_results = []

# ---------------- LOOP DATASETS ----------------
for dataset in DATASETS:
    print(f"\n========== {dataset.upper()} ==========")

    # Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, "datasets")

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    # -------- RETRIEVAL --------
    results = evaluator.retrieve(corpus, queries)

    # -------- EVALUATION --------
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, K_VALUES)

    # -------- MRR --------
    mrr = evaluator.evaluate_custom(qrels, results, K_VALUES, metric="mrr")

    print("\n--- BEFORE RERANK ---")
    print("NDCG:", ndcg)
    print("MRR:", mrr)

    # -------- RERANK --------
    print("Running re-ranking...")

    reranked_results = {}
    for qid in results:
        docs = list(results[qid].items())[:10]

        pairs = [(queries[qid], corpus[doc_id]["text"]) for doc_id, _ in docs]
        scores = reranker.predict(pairs)

        sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        reranked_results[qid] = {
            doc_id: float(score)
            for ((doc_id, _), score) in sorted_docs
        }

    # -------- EVALUATE RERANK --------
    ndcg_r, _, recall_r, precision_r = evaluator.evaluate(qrels, reranked_results, K_VALUES)
    mrr_r = evaluator.evaluate_custom(qrels, reranked_results, K_VALUES, metric="mrr")

    print("\n--- AFTER RERANK ---")
    print("NDCG:", ndcg_r)
    print("MRR:", mrr_r)

    # -------- SAVE SUMMARY --------
    summary_results.append([
        dataset,
        ndcg["NDCG@10"],
        mrr["MRR@10"],
        ndcg_r["NDCG@10"],
        mrr_r["MRR@10"]
    ])

# ---------------- SAVE CSV ----------------
with open("beir_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "dataset",
        "ndcg@10_before",
        "mrr@10_before",
        "ndcg@10_after",
        "mrr@10_after"
    ])
    writer.writerows(summary_results)

print("\nSaved: beir_summary.csv")