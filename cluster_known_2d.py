"""
Phase 3 variant — UMAP to 2D directly, cluster on that, visualize.
Compare against the 40D results from cluster_known.py.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from collections import Counter
import hdbscan
import umap

UMAP_PARAMS = {
    "n_components": 2,
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

with open("data/known_clusters_embeddings.json", encoding="utf-8") as f:
    embeddings = json.load(f)

ids          = [e["id"]                         for e in embeddings]
vectors      = np.array([e["values"]            for e in embeddings])
cluster_ids  = [e["metadata"]["cluster_id"]     for e in embeddings]
cluster_names = [e["metadata"]["cluster_name"]  for e in embeddings]
titles       = [e["metadata"]["title"]          for e in embeddings]
true_labels  = np.array(cluster_ids)

print(f"Loaded {len(ids)} embeddings ({vectors.shape[1]}-dim)")

reducer = umap.UMAP(**UMAP_PARAMS)
reduced_2d = reducer.fit_transform(vectors)
print(f"Reduced shape: {reduced_2d.shape}")

clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
labels    = clusterer.fit_predict(reduced_2d)
probs     = clusterer.probabilities_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = int((labels == -1).sum())
noise_pct  = round(100 * n_noise / len(labels), 1)

mask = labels != -1
if n_clusters >= 2 and mask.sum() > n_clusters:
    sil = round(silhouette_score(reduced_2d[mask], labels[mask]), 4)
    db  = round(davies_bouldin_score(reduced_2d[mask], labels[mask]), 4)
else:
    sil = db = "N/A"

ari = round(adjusted_rand_score(true_labels, labels), 4)
avg_conf = round(float(probs[mask].mean()), 4) if mask.sum() > 0 else "N/A"

print("\nRESULTS (2D)")
print("=" * 60)
print(f"  Clusters found     : {n_clusters}")
print(f"  Noise points       : {n_noise}  ({noise_pct}%)")
print(f"  Silhouette score   : {sil}")
print(f"  Davies-Bouldin     : {db}")
print(f"  ARI vs ground truth: {ari}")
print(f"  Avg membership conf: {avg_conf}")

for cid in sorted(set(labels)):
    mask_c = labels == cid
    gt_dist = Counter(cluster_names[i] for i in range(len(labels)) if mask_c[i])
    label_str = "NOISE" if cid == -1 else f"Cluster {cid}"
    dominant = gt_dist.most_common(1)[0][0]
    safe_dominant = dominant.encode("ascii", errors="replace").decode("ascii")
    print(f"\n  {label_str}  ({mask_c.sum()} points)  -- dominant: {safe_dominant}")

# Save results
output = {
    "umap_params":    UMAP_PARAMS,
    "hdbscan_params": {k: v for k, v in HDBSCAN_PARAMS.items() if k != "prediction_data"},
    "metrics": {
        "n_clusters": n_clusters, "noise_pct": noise_pct,
        "silhouette": sil, "davies_bouldin": db,
        "ari_vs_ground_truth": ari, "avg_membership_confidence": avg_conf,
    },
    "assignments": [
        {"pmid": ids[i], "cluster": int(labels[i]), "confidence": round(float(probs[i]), 4),
         "true_cluster": cluster_ids[i], "cluster_name": cluster_names[i], "title": titles[i]}
        for i in range(len(labels))
    ],
}

os.makedirs("data", exist_ok=True)
with open("data/known_clusters_2d_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Visualization — no second UMAP needed, reduced_2d IS the plot
PALETTE = plt.cm.tab10.colors
CLUSTER_NAMES_SHORT = {
    0: "CAR-T / B-cell lymphoma",
    1: "GLP-1 / T2 diabetes",
    2: "DBS / Parkinson's",
    3: "PD-1/PD-L1 / NSCLC",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Known-Cluster Dataset: 2D UMAP — HDBSCAN vs Ground Truth",
             fontsize=14, fontweight="bold", y=1.01)

ax = axes[0]
for lbl in sorted(set(labels)):
    m = labels == lbl
    if lbl == -1:
        ax.scatter(reduced_2d[m, 0], reduced_2d[m, 1],
                   c="lightgrey", s=20, alpha=0.6, zorder=1, label="noise")
    else:
        ax.scatter(reduced_2d[m, 0], reduced_2d[m, 1],
                   color=PALETTE[lbl % len(PALETTE)], s=30, alpha=0.85,
                   zorder=2, label=f"C{lbl}")
ax.set_title("HDBSCAN Clusters (2D)", fontsize=12, fontweight="bold")
ax.set_xlabel("UMAP-1", fontsize=9)
ax.set_ylabel("UMAP-2", fontsize=9)
ax.legend(fontsize=7, markerscale=1.3, loc="best", framealpha=0.6)
metrics_text = f"Clusters: {n_clusters}  Noise: {noise_pct}%  ARI: {ari}  Sil: {sil}"
ax.text(0.01, -0.10, metrics_text, transform=ax.transAxes, fontsize=8, color="#333333")

ax = axes[1]
for gt_id in sorted(set(cluster_ids)):
    m = true_labels == gt_id
    label = CLUSTER_NAMES_SHORT.get(gt_id, f"Cluster {gt_id}")
    ax.scatter(reduced_2d[m, 0], reduced_2d[m, 1],
               color=PALETTE[gt_id % len(PALETTE)], s=30, alpha=0.85,
               zorder=2, label=label)
ax.set_title("Ground Truth", fontsize=12, fontweight="bold")
ax.set_xlabel("UMAP-1", fontsize=9)
ax.set_ylabel("UMAP-2", fontsize=9)
ax.legend(fontsize=7, markerscale=1.3, loc="best", framealpha=0.6)

plt.tight_layout()
out_path = "data/known_clusters_2d_viz.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")
