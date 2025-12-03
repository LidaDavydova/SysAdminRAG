from typing import List
from RAG.rag_system import SysAdminRAG
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import time

# Load a small embedding model
encoder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good quality

parquet_path = "data/dataset_upload/dataset1.parquet"
df = pd.read_parquet(parquet_path)

# Sample rows for benchmark
benchmark_set = [
    {"query": row["question"], "expected": row["solution"]}
    for _, row in df.sample(100, random_state=42).iterrows()
]

def cosine_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using SentenceTransformer.
    """
    embs = encoder.encode([text1, text2], convert_to_tensor=True)
    # embs shape: [2, embedding_dim]
    return torch.nn.functional.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0)).item()


# Initialize the RAG system
rag_system = SysAdminRAG(
    ollama_url="http://localhost:11434",
    ollama_model="qwen2.5:0.5b"
)
rag_system.load_index(auto_discover=True)

class ColbertEvaluator:
    def __init__(self, rag_system):
        self.rag = rag_system

    def evaluate(self, query: str, expected_answer: str, k: int = 5) -> dict:
        """
        Evaluate a single query using RAG retrieval similarity.
        """
        results = self.rag.search(query, k=k)
        if not results:
            return {
                "query": query,
                "score_max": 0.0,
                "scores": [],
                "retrieved_texts": []
            }

        retrieved_texts = [r.get("text", "") for r in results]

        # Compute cosine similarity with expected answer
        scores = [cosine_similarity(expected_answer, t) for t in retrieved_texts]

        return {
            "query": query,
            "score_max": max(scores) if scores else 0.0,
            "scores": scores,
            "top_text": retrieved_texts[np.argmax(scores)]
        }

# Create evaluator
evaluator = ColbertEvaluator(rag_system)

results = []

for item in tqdm(benchmark_set):
    start = time.time()
    res = evaluator.evaluate(item["query"], item["expected"], k=5)
    end = time.time()

    results.append({
        "query": item["query"],
        "expected": item["expected"],
        "score_max": res["score_max"],
        "score_mean": np.mean(res["scores"]) if res["scores"] else 0.0,
        "top_text": res["top_text"],
        "retrieval_time_sec": end - start
    })

df_results = pd.DataFrame(results)
print(df_results.head())
df_results.to_csv('benchmark_rag.csv')