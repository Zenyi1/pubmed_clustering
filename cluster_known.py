"""
Phase 3 — Cluster the curated known-clusters dataset with HDBSCAN + UMAP,
store results, and produce a 2D visualization.

Reads:  data/known_clusters_embeddings.json
Writes: data/known_clusters_hdbscan_results.json
        data/known_clusters_umap_reducer.pkl
        data/known_clusters_hdbscan_clusterer.pkl
        data/known_clusters_viz.png
"""

import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from collections import Counter
import hdbscan
import umap


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
    "cluster_selection_epsilon":  0.0,
    "cluster_selection_method":   "eom",
    "metric":                    "euclidean",
    "prediction_data":           True,
}

VIZ_UMAP_PARAMS = {
    "n_components": 2,
    "n_neighbors":  5,
    "min_dist":     0.1,
    "metric":       "cosine",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open("data/known_clusters_embeddings.json", encoding="utf-8") as f:
    embeddings = json.load(f)

ids          = [e["id"]                         for e in embeddings]
vectors      = np.array([e["values"]            for e in embeddings])
cluster_ids  = [e["metadata"]["cluster_id"]     for e in embeddings]
cluster_names = [e["metadata"]["cluster_name"]  for e in embeddings]
titles       = [e["metadata"]["title"]          for e in embeddings]

true_labels = np.array(cluster_ids)

print(f"Loaded {len(ids)} embeddings ({vectors.shape[1]}-dim)")
print(f"Ground-truth clusters: {sorted(set(cluster_ids))}\n")



reducer = umap.UMAP(**UMAP_PARAMS)
reduced = reducer.fit_transform(vectors)
print(f"Reduced shape: {reduced.shape}\n")




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

ari = round(adjusted_rand_score(true_labels, labels), 4)
avg_conf = round(float(probs[mask].mean()), 4) if mask.sum() > 0 else "N/A"

print("RESULTS")
print("=" * 60)
print(f"  Clusters found     : {n_clusters}")
print(f"  Noise points       : {n_noise}  ({noise_pct}%)")
print(f"  Silhouette score   : {sil}  (higher is better, max 1)")
print(f"  Davies-Bouldin     : {db}  (lower is better)")
print(f"  ARI vs ground truth: {ari}  (1 = perfect match)")
print(f"  Avg membership conf: {avg_conf}")

# Per-cluster breakdown
cluster_label_ids = sorted(set(labels))
for cid in cluster_label_ids:
    mask_c = labels == cid
    gt_dist = Counter(
        cluster_names[i] for i in range(len(labels)) if mask_c[i]
    )
    label_str = "NOISE" if cid == -1 else f"Cluster {cid}"
    dominant = gt_dist.most_common(1)[0][0]
    safe_dominant = dominant.encode("ascii", errors="replace").decode("ascii")
    print(f"\n  {label_str}  ({mask_c.sum()} points)  -- dominant: {safe_dominant}")
    for name, count in sorted(gt_dist.items(), key=lambda x: -x[1]):
        safe_name = name.encode("ascii", errors="replace").decode("ascii")
        bar = "#" * count
        print(f"    {safe_name:<50} {count:>2}  {bar}")


output = {
    "umap_params":    UMAP_PARAMS,
    "hdbscan_params": {k: v for k, v in HDBSCAN_PARAMS.items() if k != "prediction_data"},
    "metrics": {
        "n_clusters":       n_clusters,
        "noise_pct":        noise_pct,
        "silhouette":       sil,
        "davies_bouldin":   db,
        "ari_vs_ground_truth": ari,
        "avg_membership_confidence": avg_conf,
    },
    "assignments": [
        {
            "pmid":         ids[i],
            "cluster":      int(labels[i]),
            "confidence":   round(float(probs[i]), 4),
            "true_cluster": cluster_ids[i],
            "cluster_name": cluster_names[i],
            "title":        titles[i],
        }
        for i in range(len(labels))
    ],
}

os.makedirs("data", exist_ok=True)
with open("data/known_clusters_hdbscan_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("\nSaved results to data/known_clusters_hdbscan_results.json")

# Save fitted models
with open("data/known_clusters_umap_reducer.pkl", "wb") as f:
    pickle.dump(reducer, f)
with open("data/known_clusters_hdbscan_clusterer.pkl", "wb") as f:
    pickle.dump(clusterer, f)
print("Saved UMAP reducer and HDBSCAN clusterer (.pkl)")



viz_2d = umap.UMAP(**VIZ_UMAP_PARAMS).fit_transform(vectors)

PALETTE = plt.cm.tab10.colors
CLUSTER_NAMES_SHORT = {
    0: "CAR-T / B-cell lymphoma",
    1: "GLP-1 / T2 diabetes",
    2: "DBS / Parkinson's",
    3: "PD-1/PD-L1 / NSCLC",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Known-Cluster Dataset: HDBSCAN Results vs Ground Truth",
             fontsize=14, fontweight="bold", y=1.01)

# Left: HDBSCAN clusters
ax = axes[0]
for lbl in sorted(set(labels)):
    m = labels == lbl
    if lbl == -1:
        ax.scatter(viz_2d[m, 0], viz_2d[m, 1],
                   c="lightgrey", s=20, alpha=0.6, zorder=1, label="noise")
    else:
        ax.scatter(viz_2d[m, 0], viz_2d[m, 1],
                   color=PALETTE[lbl % len(PALETTE)], s=30, alpha=0.85,
                   zorder=2, label=f"C{lbl}")
ax.set_title("HDBSCAN Clusters", fontsize=12, fontweight="bold")
ax.set_xlabel("UMAP-1", fontsize=9)
ax.set_ylabel("UMAP-2", fontsize=9)
ax.legend(fontsize=7, markerscale=1.3, loc="best", framealpha=0.6)
metrics_text = (f"Clusters: {n_clusters}  Noise: {noise_pct}%  "
                f"ARI: {ari}  Sil: {sil}")
ax.text(0.01, -0.10, metrics_text, transform=ax.transAxes,
        fontsize=8, color="#333333")

# Right: ground truth
ax = axes[1]
for gt_id in sorted(set(cluster_ids)):
    m = true_labels == gt_id
    label = CLUSTER_NAMES_SHORT.get(gt_id, f"Cluster {gt_id}")
    ax.scatter(viz_2d[m, 0], viz_2d[m, 1],
               color=PALETTE[gt_id % len(PALETTE)], s=30, alpha=0.85,
               zorder=2, label=label)
ax.set_title("Ground Truth", fontsize=12, fontweight="bold")
ax.set_xlabel("UMAP-1", fontsize=9)
ax.set_ylabel("UMAP-2", fontsize=9)
ax.legend(fontsize=7, markerscale=1.3, loc="best", framealpha=0.6)

plt.tight_layout()
out_path = "data/known_clusters_viz.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization to {out_path}")
