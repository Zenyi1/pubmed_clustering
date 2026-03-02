"""
Hyperparameter sweep on the Phase 2 dataset (10 broad medical categories).
Same grid as the known-cluster sweep for direct comparison.
"""

import json
import itertools
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import hdbscan
import umap

UMAP_N_COMPONENTS = [2, 5, 10, 20, 40, 60]
UMAP_N_NEIGHBORS  = [3, 5, 10, 15]
UMAP_MIN_DIST     = [0.0, 0.05, 0.1]

HDBSCAN_MIN_CLUSTER_SIZE = [3, 5, 8, 10]
HDBSCAN_METHOD           = ["eom", "leaf"]

with open("data/embeddings.json", encoding="utf-8") as f:
    embeddings = json.load(f)

vectors    = np.array([e["values"]            for e in embeddings])
categories = [e["metadata"]["category"]       for e in embeddings]

le          = LabelEncoder()
true_labels = le.fit_transform(categories)

print(f"Loaded {len(embeddings)} embeddings ({vectors.shape[1]}-dim)")
print(f"Categories: {list(le.classes_)}\n")

combos = list(itertools.product(
    UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_METHOD,
))
print(f"Total combinations: {len(combos)}\n")

umap_cache = {}
results = []

for i, (n_comp, n_neigh, min_dist, min_cs, method) in enumerate(combos):
    umap_key = (n_comp, n_neigh, min_dist)
    if umap_key not in umap_cache:
        reducer = umap.UMAP(
            n_components=n_comp, n_neighbors=n_neigh, min_dist=min_dist,
            metric="cosine", random_state=42,
        )
        umap_cache[umap_key] = reducer.fit_transform(vectors)

    reduced = umap_cache[umap_key]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cs, cluster_selection_method=method,
        metric="euclidean", prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    probs  = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    noise_pct  = round(100 * n_noise / len(labels), 1)

    mask = labels != -1
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = round(silhouette_score(reduced[mask], labels[mask]), 4)
        db  = round(davies_bouldin_score(reduced[mask], labels[mask]), 4)
    else:
        sil = 0.0
        db  = 99.0

    ari = round(adjusted_rand_score(true_labels, labels), 4)
    avg_conf = round(float(probs[mask].mean()), 4) if mask.sum() > 0 else 0.0

    results.append({
        "umap_dims": n_comp, "umap_neighbors": n_neigh, "umap_min_dist": min_dist,
        "hdbscan_min_cs": min_cs, "hdbscan_method": method,
        "clusters": n_clusters, "noise_pct": noise_pct,
        "silhouette": sil, "davies_bouldin": db,
        "ari": ari, "avg_conf": avg_conf,
    })

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(combos)} done...")

results.sort(key=lambda r: (-r["ari"], -r["silhouette"], r["noise_pct"]))

print(f"\n{'=' * 120}")
print(f"{'dims':>4}  {'neigh':>5}  {'mdist':>5}  {'mcs':>3}  {'method':>5}  "
      f"{'clust':>5}  {'noise%':>6}  {'ARI':>7}  {'Sil':>7}  {'DB':>7}  {'conf':>6}")
print(f"{'=' * 120}")

for r in results[:30]:
    print(f"{r['umap_dims']:>4}  {r['umap_neighbors']:>5}  {r['umap_min_dist']:>5.2f}  "
          f"{r['hdbscan_min_cs']:>3}  {r['hdbscan_method']:>5}  "
          f"{r['clusters']:>5}  {r['noise_pct']:>5.1f}%  {r['ari']:>7.4f}  "
          f"{r['silhouette']:>7.4f}  {r['davies_bouldin']:>7.4f}  {r['avg_conf']:>6.4f}")

# Also show best 2D results specifically
print(f"\n--- Best 2D results ---")
twod = [r for r in results if r["umap_dims"] == 2]
for r in twod[:10]:
    print(f"  neigh={r['umap_neighbors']:>2}  mdist={r['umap_min_dist']:.2f}  "
          f"mcs={r['hdbscan_min_cs']:>2}  {r['hdbscan_method']:>5}  "
          f"clust={r['clusters']:>2}  noise={r['noise_pct']:>5.1f}%  "
          f"ARI={r['ari']:.4f}  Sil={r['silhouette']:.4f}")

with open("data/hyperparam_sweep_phase2_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nSaved to data/hyperparam_sweep_phase2_results.json")
