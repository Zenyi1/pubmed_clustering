import json
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

EMBEDDING_MODEL = "multilingual-e5-large"
BATCH_SIZE = 10  # number of abstracts to embed per API call
ABSTRACT_METADATA_LIMIT = 500  # chars stored in Pinecone metadata

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

with open("data/articles.json", encoding="utf-8") as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles. Embedding in batches of {BATCH_SIZE}...")

embeddings = []

for i in range(0, len(articles), BATCH_SIZE):
    batch = articles[i : i + BATCH_SIZE]
    abstracts = [a["abstract"] for a in batch]

    result = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=abstracts,
        parameters={"input_type": "passage"},
    )

    for article, embedding in zip(batch, result):
        embeddings.append({
            "id": article["pmid"],
            "values": embedding.values,
            "metadata": {
                "pmid":            article["pmid"],
                "title":           article["title"],
                "abstract":        article["abstract"][:ABSTRACT_METADATA_LIMIT],
                "category":        article["category"],
                "url":             article["url"],
                "relevance_score": article["relevance_score"],
            },
        })

    print(f"  Embedded {min(i + BATCH_SIZE, len(articles))}/{len(articles)}")
    time.sleep(0.3)  # brief pause between batches

os.makedirs("data", exist_ok=True)
with open("data/embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, ensure_ascii=False)

# Spot-check
sample = embeddings[0]
print(f"\nSpot-check on first vector:")
print(f"  id:              {sample['id']}")
print(f"  vector length:   {len(sample['values'])}")
print(f"  category:        {sample['metadata']['category']}")
print(f"  abstract[:80]:   {sample['metadata']['abstract'][:80]}...")
print(f"\nTotal embeddings saved: {len(embeddings)}")
print("Saved to data/embeddings.json")
