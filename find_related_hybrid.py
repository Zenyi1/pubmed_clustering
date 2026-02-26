"""
Find related papers using hybrid (dense + sparse) Pinecone search.
Predicts which cluster a paper would fall into via majority vote from neighbors.

Usage:
  python find_related_hybrid.py 38844093                          # lookup by PMID
  python find_related_hybrid.py "heart failure treatment outcomes" # search by text
"""

import json
import os
import sys
from collections import Counter
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "pubmed-hybrid"
DENSE_MODEL = "multilingual-e5-large"
SPARSE_MODEL = "pinecone-sparse-english-v0"
TOP_K = 5

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# Load articles for PMID -> abstract lookup
with open("data/articles.json", encoding="utf-8") as f:
    articles = json.load(f)
articles_by_pmid = {a["pmid"]: a for a in articles}

# Load cluster assignments for majority vote
with open("data/hdbscan_results.json", encoding="utf-8") as f:
    cluster_data = json.load(f)
cluster_by_pmid = {a["pmid"]: a for a in cluster_data["assignments"]}


def embed_text(text):
    """Embed text with both dense and sparse models."""
    dense = pc.inference.embed(
        model=DENSE_MODEL,
        inputs=[text],
        parameters={"input_type": "query"},
    )
    sparse = pc.inference.embed(
        model=SPARSE_MODEL,
        inputs=[text],
        parameters={"input_type": "query"},
    )
    return dense[0], sparse[0]


def query_hybrid(dense_emb, sparse_emb, top_k=TOP_K, exclude_id=None):
    """Query hybrid index with both dense and sparse vectors."""
    k = top_k + (1 if exclude_id else 0)

    results = index.query(
        vector=dense_emb.values,
        sparse_vector={
            "indices": list(sparse_emb.sparse_indices),
            "values": list(sparse_emb.sparse_values),
        },
        top_k=k,
        include_metadata=True,
    )

    matches = results.matches
    if exclude_id:
        matches = [m for m in matches if m.id != exclude_id]
    return matches[:top_k]


def predict_cluster(matches):
    """Predict cluster via majority vote from neighbor cluster assignments."""
    votes = []
    for m in matches:
        if m.id in cluster_by_pmid:
            cid = cluster_by_pmid[m.id]["cluster"]
            if cid != -1:  # skip noise
                votes.append(cid)

    if not votes:
        return -1, {}

    vote_counts = Counter(votes)
    predicted = vote_counts.most_common(1)[0][0]
    return predicted, dict(vote_counts.most_common())


def print_results(query_label, matches, predicted_cluster, vote_dist):
    """Print formatted results."""
    print(f"\nQuery: {query_label}")
    print("=" * 80)

    if predicted_cluster == -1:
        print("  Predicted cluster: NONE (all neighbors are noise)")
    else:
        print(f"  Predicted cluster: {predicted_cluster}")
        vote_str = ", ".join(f"cluster {c}: {n} votes" for c, n in vote_dist.items())
        print(f"  Vote distribution: {vote_str}")

    print(f"\n  Top {len(matches)} matches:")
    print(f"  {'Score':>8}  {'Cluster':>8}  {'Category':<22}  Title")
    print(f"  {'-'*8}  {'-'*8}  {'-'*22}  {'-'*40}")

    for m in matches:
        cat = m.metadata.get("category", "?")
        title = m.metadata.get("title", "?")[:60]
        cinfo = cluster_by_pmid.get(m.id)
        cid = cinfo["cluster"] if cinfo else "?"
        cid_str = "NOISE" if cid == -1 else str(cid)
        print(f"  {m.score:>8.4f}  {cid_str:>8}  {cat:<22}  {title}")


def run(query_input):
    """Run a hybrid search given a PMID or raw text."""
    exclude_id = None

    if query_input in articles_by_pmid:
        # PMID mode
        article = articles_by_pmid[query_input]
        text = article["abstract"]
        label = f"PMID {query_input} — {article['title']}"
        exclude_id = query_input

        own_cluster = cluster_by_pmid.get(query_input)
        if own_cluster:
            cid = own_cluster["cluster"]
            print(f"  (Actual cluster: {cid}{'  NOISE' if cid == -1 else ''}, "
                  f"category: {own_cluster['category']})")
    else:
        # Raw text mode
        text = query_input
        label = f'"{query_input[:70]}"'

    dense_emb, sparse_emb = embed_text(text)
    matches = query_hybrid(dense_emb, sparse_emb, exclude_id=exclude_id)
    predicted, vote_dist = predict_cluster(matches)
    print_results(label, matches, predicted, vote_dist)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        # Demo: run one PMID and one text query
        print("=== Demo: PMID lookup ===")
        run("38844093")
        print("\n\n=== Demo: Free text ===")
        run("heart failure treatment outcomes")
