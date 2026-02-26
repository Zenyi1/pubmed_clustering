import json
import os
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import hdbscan
import umap


'''
i've been playing around with these, the most important ones are n_components in UMAP params -> how small do we want to make the original 1024 vector
the other one is min_cluster_size in HDBSCAN params -> how small do we want the clusters to be (try 3–15, smaller = more clusters, larger = fewer)

I am convinced HDBSCAN is the best clustering at least for dense vectors.
'''

USE_UMAP = True  # False = run HDBSCAN directly on raw 1024-dim vectors (no reduction)

UMAP_PARAMS = {
    "n_components": 40,       # target dimensions -> orginal vector is 1024
    "n_neighbors":  5,       # local vs global structure, small is just thight clusters which we want
    "min_dist":     0.0,      # 0.0 keeps points tightly packed (best for clustering)
    "metric":       "cosine", # standard for vector stuff
    "random_state": 42, }

HDBSCAN_PARAMS = {
    "min_cluster_size":       3,       # smallest cluster allowed 
    "min_samples":            None,    # None = same as min_cluster_size
    "cluster_selection_epsilon": 0.0,  # not needed, would merge small nearby clusters
    "cluster_selection_method":  "eom", # "eom" (excess of mass, default) or "leaf" gives smaller clusters
    "metric":                "euclidean", # we use UMAP already with cosine so we do euclidean
    #"algorithm":             "generic",   failed experiment
    "prediction_data":       True,     # enables soft membership scores more on that below
}


with open("data/embeddings.json", encoding="utf-8") as f:
    embeddings = json.load(f)

ids        = [e["id"]                    for e in embeddings]
vectors    = np.array([e["values"]       for e in embeddings])
categories = [e["metadata"]["category"] for e in embeddings]
titles     = [e["metadata"]["title"]    for e in embeddings]

le          = LabelEncoder()
true_labels = le.fit_transform(categories)


if USE_UMAP:
    reducer = umap.UMAP(**UMAP_PARAMS)
    reduced = reducer.fit_transform(vectors)
    print(f"Reduced shape: {reduced.shape}\n")
else:
    reduced = vectors


clusterer   = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
labels      = clusterer.fit_predict(reduced)
probs       = clusterer.probabilities_   # soft membership confidence per point -> how likely this point belongs to a cluster, useful if some medical papers fall in many categories

#Metrics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = int((labels == -1).sum())
noise_pct  = round(100 * n_noise / len(labels), 1)

mask = labels != -1
if n_clusters >= 2 and mask.sum() > n_clusters:
    sil = round(silhouette_score(reduced[mask], labels[mask]), 4)
    db  = round(davies_bouldin_score(reduced[mask], labels[mask]), 4)
else:
    sil = db = "N/A"

ari = round(adjusted_rand_score(true_labels, labels), 4)
avg_conf = round(float(probs[mask].mean()), 4) if mask.sum() > 0 else "N/A"


#claude made this, explanantion is next to each score
print("RESULTS")
print("=" * 60)
print(f"  Clusters found     : {n_clusters}")
print(f"  Noise points       : {n_noise}  ({noise_pct}%)")
print(f"  Silhouette score   : {sil}  (higher is better, max 1)")
print(f"  Davies-Bouldin     : {db}  (lower is better)")
print(f"  ARI vs categories  : {ari}  (1 = perfect match to known labels)")
print(f"  Avg membership conf: {avg_conf}  (how certain each point is in its cluster)")


#visualizing categoies in each cluster, also claude, you can verify this in hdbscan results within data
cluster_ids = sorted(set(labels))
for cid in cluster_ids:
    mask_c   = labels == cid
    cats     = [categories[i] for i in range(len(labels)) if mask_c[i]]
    cat_dist = Counter(cats)
    label    = "NOISE" if cid == -1 else f"Cluster {cid}"
    dominant = cat_dist.most_common(1)[0][0]
    print(f"\n  {label}  ({mask_c.sum()} points)  — dominant: {dominant}")
    for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"    {cat:<22} {count:>2}  {bar}")

#save the results to json
output = {
    "umap_params":    UMAP_PARAMS,
    "hdbscan_params": {k: v for k, v in HDBSCAN_PARAMS.items() if k != "prediction_data"},
    "metrics": {
        "n_clusters":       n_clusters,
        "noise_pct":        noise_pct,
        "silhouette":       sil,
        "davies_bouldin":   db,
        "ari_vs_categories": ari,
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
with open("data/hdbscan_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
