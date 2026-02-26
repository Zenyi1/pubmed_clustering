'''
Claude code created this, I use umap to make it 2 dimensional and plot it
'''


import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
from scipy.sparse import csr_matrix

with open("data/embeddings.json", encoding="utf-8") as f:
    embeddings = json.load(f)

with open("data/hdbscan_results.json", encoding="utf-8") as f:
    dense_results = json.load(f)

dense_vectors = np.array([e["values"] for e in embeddings])
dense_labels  = np.array([a["cluster"]  for a in dense_results["assignments"]])
categories    = [a["category"] for a in dense_results["assignments"]]


with open("data/sparse_embeddings.json", encoding="utf-8") as f:
    sparse_data = json.load(f)

with open("data/hdbscan_sparse_results.json", encoding="utf-8") as f:
    sparse_results = json.load(f)

sparse_labels = np.array([a["cluster"] for a in sparse_results["assignments"]])

# Rebuild sparse matrix (same hash remapping as cluster_sparse.py)
all_hashes  = set()
for doc in sparse_data:
    all_hashes.update(doc["sparse_indices"])
hash_to_col = {h: i for i, h in enumerate(sorted(all_hashes))}

rows, cols, data_vals = [], [], []
for doc_idx, doc in enumerate(sparse_data):
    for h, v in zip(doc["sparse_indices"], doc["sparse_values"]):
        rows.append(doc_idx)
        cols.append(hash_to_col[h])
        data_vals.append(v)

sparse_matrix = csr_matrix(
    (data_vals, (rows, cols)),
    shape=(len(sparse_data), len(hash_to_col)),
)

VIZ_PARAMS = {
    "n_components": 2,
    "n_neighbors":  5,
    "min_dist":     0.1,   # slightly spread out for readability
    "metric":       "cosine",
    "random_state": 42,
}

print("Computing 2D UMAP for dense vectors...")
dense_2d  = umap.UMAP(**VIZ_PARAMS).fit_transform(dense_vectors)

print("Computing 2D UMAP for sparse matrix...")
sparse_2d = umap.UMAP(**VIZ_PARAMS).fit_transform(sparse_matrix)

CLUSTER_PALETTE  = plt.cm.tab20.colors
CATEGORY_PALETTE = plt.cm.tab10.colors

unique_cats = sorted(set(categories))
cat_to_int  = {c: i for i, c in enumerate(unique_cats)}
cat_ints    = np.array([cat_to_int[c] for c in categories])

def scatter_by_label(ax, xy, labels, palette, title):
    """Color points by integer label. Label -1 = noise (grey)."""
    unique = sorted(set(labels))
    for lbl in unique:
        mask  = labels == lbl
        if lbl == -1:
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c="lightgrey", s=18, alpha=0.6, zorder=1, label="noise")
        else:
            color = palette[lbl % len(palette)]
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       color=color, s=28, alpha=0.85, zorder=2, label=f"C{lbl}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ncol = max(1, len(unique) // 8)
    ax.legend(fontsize=6, markerscale=1.4, loc="best", ncol=ncol,
              framealpha=0.6, handletextpad=0.3, borderpad=0.4)

def scatter_by_category(ax, xy, cat_labels, unique_cats, palette, title):
    """Color points by medical category."""
    for i, cat in enumerate(unique_cats):
        mask  = np.array([c == cat for c in cat_labels])
        color = palette[i % len(palette)]
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   color=color, s=28, alpha=0.85, zorder=2, label=cat)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, markerscale=1.4, loc="best", ncol=1,
              framealpha=0.6, handletextpad=0.3, borderpad=0.4)

dm = dense_results["metrics"]
sm = sparse_results["metrics"]

dense_ann  = (f"Clusters: {dm['n_clusters']}  Noise: {dm['noise_pct']}%  "
              f"ARI: {dm['ari_vs_categories']}  Sil: {dm['silhouette']}")
sparse_ann = (f"Clusters: {sm['n_clusters']}  Noise: {sm['noise_pct']}%  "
              f"ARI: {sm['ari_vs_categories']}  Sil: {sm['silhouette']}")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Dense vs Sparse Embedding Clustering", fontsize=14, fontweight="bold", y=0.99)

scatter_by_label(    axes[0, 0], dense_2d,  dense_labels,  CLUSTER_PALETTE,  "Dense — by Cluster")
scatter_by_category( axes[0, 1], dense_2d,  categories,    unique_cats, CATEGORY_PALETTE, "Dense — by Category")
scatter_by_label(    axes[1, 0], sparse_2d, sparse_labels, CLUSTER_PALETTE,  "Sparse — by Cluster")
scatter_by_category( axes[1, 1], sparse_2d, categories,    unique_cats, CATEGORY_PALETTE, "Sparse — by Category")

# Metrics sub-labels below each row title
axes[0, 0].text(0.01, -0.14, dense_ann,  transform=axes[0, 0].transAxes,
                fontsize=7.5, color="#333333")
axes[1, 0].text(0.01, -0.14, sparse_ann, transform=axes[1, 0].transAxes,
                fontsize=7.5, color="#333333")

plt.tight_layout(rect=[0, 0, 1, 0.98])

out_path = "data/cluster_comparison.png"
os.makedirs("data", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")
