import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "pubmed-abstracts"
EMBEDDING_MODEL = "multilingual-e5-large"
DIMENSION = 1024
METRIC = "cosine"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

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
stats = index.describe_index_stats()
print(f"\nIndex stats:")
print(f"  name:      {INDEX_NAME}")
print(f"  dimension: {DIMENSION}")
print(f"  metric:    {METRIC}")
print(f"  model:     {EMBEDDING_MODEL}")
print(f"  vectors:   {stats.total_vector_count}")
