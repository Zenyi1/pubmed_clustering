"""
Predict which HDBSCAN cluster a new paper belongs to using the fitted UMAP + HDBSCAN models.
Embeds text via Pinecone API, transforms through saved UMAP reducer, then uses
HDBSCAN approximate_predict to get a cluster label and confidence score.

Requires: run hdbscan_cluster.py first to generate the .pkl model files.

Usage:
  python predict_cluster.py "heart failure treatment outcomes" # predict from free text
"""

import json
import os
import pickle
import sys
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from hdbscan import approximate_predict

load_dotenv()

DENSE_MODEL = "multilingual-e5-large"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Load cluster results for showing cluster composition
with open("data/hdbscan_results.json", encoding="utf-8") as f:
    cluster_data = json.load(f)

# Load fitted models
with open("data/umap_reducer.pkl", "rb") as f:
    reducer = pickle.load(f)

with open("data/hdbscan_clusterer.pkl", "rb") as f:
    clusterer = pickle.load(f)

# Build cluster composition lookup for context
clusters_composition = {}
for a in cluster_data["assignments"]:
    cid = a["cluster"]
    if cid == -1:
        continue
    if cid not in clusters_composition:
        clusters_composition[cid] = []
    clusters_composition[cid].append(a)


def embed_dense(text):
    """Embed text with dense model via Pinecone inference API."""
    result = pc.inference.embed(
        model=DENSE_MODEL,
        inputs=[text],
        parameters={"input_type": "query"},
    )
    return np.array(result[0].values).reshape(1, -1)


def predict(text):
    """Embed, reduce via UMAP, predict via HDBSCAN approximate_predict."""
    vector = embed_dense(text)

    # Transform through fitted UMAP (same projection used during clustering)
    reduced = reducer.transform(vector)

    # approximate_predict uses the HDBSCAN density landscape to place the new point
    pred_labels, pred_strengths = approximate_predict(clusterer, reduced)
    pred_cluster = int(pred_labels[0])
    pred_confidence = round(float(pred_strengths[0]), 4)

    # Print results
    print(f"\nQuery: \"{text[:70]}\"")
    print("=" * 80)

    pred_str = "NOISE" if pred_cluster == -1 else str(pred_cluster)
    print(f"  Predicted cluster: {pred_str}")
    print(f"  Confidence:        {pred_confidence}")

    # Show what's in the predicted cluster
    if pred_cluster != -1 and pred_cluster in clusters_composition:
        papers = clusters_composition[pred_cluster]
        cats = {}
        for p in papers:
            cats[p["category"]] = cats.get(p["category"], 0) + 1
        cat_str = ", ".join(f"{c}({n})" for c, n in sorted(cats.items(), key=lambda x: -x[1]))
        print(f"  Cluster {pred_cluster} contains: {len(papers)} papers -- {cat_str}")
        for p in sorted(papers, key=lambda x: -x["confidence"])[:5]:
            print(f"    [{p['confidence']:.4f}] {p['category']:<22} {p['title'][:60]}")
        if len(papers) > 5:
            print(f"    ... and {len(papers) - 5} more")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict_cluster.py \"your abstract text here\"")
