#same as densie with some extra steps, pinecone does not return the fixed length vector but instead gives you the indices that are nonzero. This is for all possible tokens 
#to fix it reduce to the amount of unique tokens across our corpus which is 4057, this number will be a lot greater on the larger full pubmed corpus, still cant be huge cause medical vocabulary has a finite ceiling surely smaller than all the 4 bilion possibilities

import json
import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from scipy.sparse import csr_matrix
import hdbscan
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

load_dotenv()

UMAP_PARAMS = {
    "n_components": 40,
    "n_neighbors":  5,
    "min_dist":     0.0,
    "metric":       "cosine",
    "random_state": 42,
}

HDBSCAN_PARAMS = {
    "min_cluster_size":          3,
    "min_samples":               None,
    "cluster_selection_epsilon": 0.0,
    "cluster_selection_method":  "eom",
    "metric":                    "euclidean",
    "prediction_data":           True,
}

SPARSE_CACHE = "data/sparse_embeddings.json"
BATCH_SIZE   = 10

with open("data/articles.json", encoding="utf-8") as f:
    articles = json.load(f)

ids        = [a["pmid"]     for a in articles]
categories = [a["category"] for a in articles]
titles     = [a["title"]    for a in articles]
abstracts  = [a["abstract"] for a in articles]

le          = LabelEncoder()
true_labels = le.fit_transform(categories)

if os.path.exists(SPARSE_CACHE):
    print("Loading cached sparse embeddings...")
    with open(SPARSE_CACHE, encoding="utf-8") as f:
        sparse_data = json.load(f)
else:
    print("Generating sparse embeddings via Pinecone (pinecone-sparse-english-v0)...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    sparse_data = []

    for i in range(0, len(abstracts), BATCH_SIZE):
        batch_texts  = abstracts[i:i + BATCH_SIZE]
        batch_ids    = ids[i:i + BATCH_SIZE]
        batch_cats   = categories[i:i + BATCH_SIZE]
        batch_titles = titles[i:i + BATCH_SIZE]

        response = pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=batch_texts,
            parameters={"input_type": "passage", "return_tokens": False},
        )

        for j, emb in enumerate(response):
            # sparse_indices: large hash IDs, sparse_values: float weights
            sparse_data.append({
                "id":             batch_ids[j],
                "sparse_indices": list(emb.sparse_indices),
                "sparse_values":  list(emb.sparse_values),
                "category":       batch_cats[j],
                "title":          batch_titles[j],
            })

        print(f"  Embedded {min(i + BATCH_SIZE, len(abstracts))}/{len(abstracts)}")

    os.makedirs("data", exist_ok=True)
    with open(SPARSE_CACHE, "w", encoding="utf-8") as f:
        json.dump(sparse_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {SPARSE_CACHE}")


#this gives us the correct vector structure we need
all_hashes = set()
for doc in sparse_data:
    all_hashes.update(doc["sparse_indices"])

hash_to_col = {h: i for i, h in enumerate(sorted(all_hashes))}
n_terms     = len(hash_to_col)

rows, cols, data_vals = [], [], []
for doc_idx, doc in enumerate(sparse_data):
    for h, v in zip(doc["sparse_indices"], doc["sparse_values"]):
        rows.append(doc_idx)
        cols.append(hash_to_col[h])
        data_vals.append(v)

sparse_matrix = csr_matrix(
    (data_vals, (rows, cols)),
    shape=(len(sparse_data), n_terms),
)

avg_nnz = sparse_matrix.nnz / sparse_matrix.shape[0]
print(f"Matrix: {sparse_matrix.shape[0]} docs x {n_terms} unique tokens")
print(f"Avg non-zero tokens per doc: {avg_nnz:.1f}")

print("\nRunning UMAP...")
reducer = umap.UMAP(**UMAP_PARAMS)
reduced = reducer.fit_transform(sparse_matrix)
print(f"Reduced shape: {reduced.shape}")

print("Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
labels    = clusterer.fit_predict(reduced)
probs     = clusterer.probabilities_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = int((labels == -1).sum())
noise_pct  = round(100 * n_noise / len(labels), 1)

mask = labels != -1
if n_clusters >= 2 and mask.sum() > n_clusters:
    sil = round(silhouette_score(reduced[mask], labels[mask]), 4)
    db  = round(davies_bouldin_score(reduced[mask], labels[mask]), 4)
else:
    sil = db = "N/A"

ari      = round(adjusted_rand_score(true_labels, labels), 4)
avg_conf = round(float(probs[mask].mean()), 4) if mask.sum() > 0 else "N/A"

print("\nSPARSE RESULTS (pinecone-sparse-english-v0 + UMAP + HDBSCAN)")
print("=" * 60)
print(f"  Clusters found     : {n_clusters}")
print(f"  Noise points       : {n_noise}  ({noise_pct}%)")
print(f"  Silhouette score   : {sil}  (higher is better, max 1)")
print(f"  Davies-Bouldin     : {db}  (lower is better)")
print(f"  ARI vs categories  : {ari}  (1 = perfect match to known labels)")
print(f"  Avg membership conf: {avg_conf}  (how certain each point is in its cluster)")

# Cluster composition
cluster_ids = sorted(set(labels))
for cid in cluster_ids:
    mask_c   = labels == cid
    cats     = [categories[i] for i in range(len(labels)) if mask_c[i]]
    cat_dist = Counter(cats)
    label    = "NOISE" if cid == -1 else f"Cluster {cid}"
    dominant = cat_dist.most_common(1)[0][0]
    print(f"\n  {label}  ({mask_c.sum()} points)  - dominant: {dominant}")
    for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"    {cat:<22} {count:>2}  {bar}")

print("\nCOMPARISON vs DENSE (multilingual-e5-large)")
print("=" * 60)
try:
    with open("data/hdbscan_results.json", encoding="utf-8") as f:
        dense = json.load(f)
    dm = dense["metrics"]
    print(f"{'Metric':<30} {'Dense':>10} {'Sparse':>10}")
    print("-" * 52)
    print(f"{'Clusters found':<30} {dm['n_clusters']:>10} {n_clusters:>10}")
    print(f"{'Noise %':<30} {str(dm['noise_pct'])+'%':>10} {str(noise_pct)+'%':>10}")
    print(f"{'Silhouette':<30} {str(dm['silhouette']):>10} {str(sil):>10}")
    print(f"{'Davies-Bouldin':<30} {str(dm['davies_bouldin']):>10} {str(db):>10}")
    print(f"{'ARI':<30} {str(dm['ari_vs_categories']):>10} {str(ari):>10}")
    print(f"{'Avg confidence':<30} {str(dm['avg_membership_confidence']):>10} {str(avg_conf):>10}")
except FileNotFoundError:
    print("  data/hdbscan_results.json not found — run hdbscan_cluster.py first")

output = {
    "embedding_type": "sparse",
    "model":          "pinecone-sparse-english-v0",
    "umap_params":    UMAP_PARAMS,
    "hdbscan_params": {k: v for k, v in HDBSCAN_PARAMS.items() if k != "prediction_data"},
    "metrics": {
        "n_clusters":                n_clusters,
        "noise_pct":                 noise_pct,
        "silhouette":                sil,
        "davies_bouldin":            db,
        "ari_vs_categories":         ari,
        "avg_membership_confidence": avg_conf,
    },
    "assignments": [
        {
            "pmid":       ids[i],
            "cluster":    int(labels[i]),
            "confidence": round(float(probs[i]), 4),
            "category":   categories[i],
            "title":      titles[i],
        }
        for i in range(len(labels))
    ],
}

os.makedirs("data", exist_ok=True)
with open("data/hdbscan_sparse_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
