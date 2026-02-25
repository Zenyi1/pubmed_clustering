import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "pubmed-abstracts"
BATCH_SIZE = 50

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

with open("data/embeddings.json", encoding="utf-8") as f:
    embeddings = json.load(f)

print(f"Loaded {len(embeddings)} vectors. Upserting in batches of {BATCH_SIZE}...")

for i in range(0, len(embeddings), BATCH_SIZE):
    batch = embeddings[i : i + BATCH_SIZE]
    vectors = [{"id": e["id"], "values": e["values"], "metadata": e["metadata"]} for e in batch]
    index.upsert(vectors=vectors)
    print(f"  Upserted {min(i + BATCH_SIZE, len(embeddings))}/{len(embeddings)}")

stats = index.describe_index_stats()
print(f"\nFinal index stats:")
print(f"  total vectors: {stats.total_vector_count}")

by_category = {}
for e in embeddings:
    cat = e["metadata"]["category"]
    by_category.setdefault(cat, 0)
    by_category[cat] += 1

print(f"\nVectors by category:")
for cat, count in by_category.items():
    print(f"  {cat}: {count}")
