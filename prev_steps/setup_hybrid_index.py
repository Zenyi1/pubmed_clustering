"""
Create hybrid Pinecone index (dense + sparse) and upsert all 100 papers.
Dense: multilingual-e5-large (1024d), Sparse: pinecone-sparse-english-v0 (SPLADE)
Metric must be dotproduct for hybrid queries.
"""

import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "pubmed-hybrid"
DIMENSION = 1024
METRIC = "dotproduct"
BATCH_SIZE = 50

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# --- Create index if needed ---
existing = [idx.name for idx in pc.list_indexes()]

if INDEX_NAME in existing:
    print(f"Index '{INDEX_NAME}' already exists, skipping creation.")
else:
    print(f"Creating index '{INDEX_NAME}' ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created.")

index = pc.Index(INDEX_NAME)

# --- Load dense + sparse embeddings ---
print("\nLoading embeddings...")
with open("data/embeddings.json", encoding="utf-8") as f:
    dense_data = json.load(f)

with open("data/sparse_embeddings.json", encoding="utf-8") as f:
    sparse_data = json.load(f)

# Build sparse lookup by PMID
sparse_by_id = {s["id"]: s for s in sparse_data}

# --- Upsert in batches ---
vectors = []
for d in dense_data:
    pmid = d["id"]
    s = sparse_by_id[pmid]

    vectors.append({
        "id": pmid,
        "values": d["values"],
        "sparse_values": {
            "indices": s["sparse_indices"],
            "values": s["sparse_values"],
        },
        "metadata": d["metadata"],
    })

print(f"Upserting {len(vectors)} vectors in batches of {BATCH_SIZE}...")
for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i + BATCH_SIZE]
    index.upsert(vectors=batch)
    print(f"  Upserted {min(i + BATCH_SIZE, len(vectors))}/{len(vectors)}")

# --- Verify ---
stats = index.describe_index_stats()
print(f"\nIndex stats:")
print(f"  name:      {INDEX_NAME}")
print(f"  dimension: {DIMENSION}")
print(f"  metric:    {METRIC}")
print(f"  vectors:   {stats.total_vector_count}")
