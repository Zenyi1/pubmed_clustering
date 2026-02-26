# PubMed Clustering

Fetches PubMed abstracts via the Valyu API, embeds them using Pinecone's inference API, and runs HDBSCAN density-based clustering with UMAP dimensionality reduction.

The vectors are inside the embedding.json so you don't have to do anything inside the prev_step folder, unless you want to change the original data. You can just run hdbscan which is heavily commented

For the test of clustering I have 100 pubmed papers across 10 medical categories (cardiology, pulmonology, nephrology, etc...). I tried dbscan and optics but HDBSCAN was the clear winner for this. I perform both dense and sparse clustering on both.

Contrary to my research I HAD to do dimensionality reduction for the clustering to work. The results were not great and incredibly volatile (chaning 1 unit of a hyperparameter broke everything or kept it the same). More on that decision below ->



## Requirements

- Python 3.12
- A `.env` file in the project root with:

```
PINECONE_API_KEY=...
VALYU_API_KEY=...
```

## Setup

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Running the clustering

```bash
venv/Scripts/python.exe hdbscan_cluster.py
```

Output:
- Metrics table printed to terminal (clusters found, noise %, silhouette, Davies-Bouldin, ARI) 
#### Scores are not super important, because the grouping per defined category (all cardiology papers together) is not necessarily correct, e.g. multidisciplinary papers or different topics within a field should be clustered separately so that is why the json results are really good to check.

- Cluster composition breakdown by medical category
- Low-confidence assignments flagged
- Results saved to `data/hdbscan_results.json`

## Tuning

Open `hdbscan_cluster.py` and edit the parameter blocks at the top.
The sae is done with sparse vectors in `cluster_sparse.py` although some preprocessing is done to work efficiently with sparse vectors given they are returned in a very unique data structure.

**UMAP**

| Parameter | Default | Effect |
|---|---|---|
| `USE_UMAP` | `True` | Set `False` to skip reduction and run on raw 1024-dim vectors |
| `n_components` | `40` | Target dimensions after reduction. Lower = more compression |
| `n_neighbors` | `5` | Small = tight local clusters. Large = broader global shape |
| `min_dist` | `0.0` | Keep at 0.0 for clustering. Higher values spread points out |

**HDBSCAN**

| Parameter | Default | Effect |
|---|---|---|
| `min_cluster_size` | `3` | Smallest allowed cluster. Lower = more clusters |
| `min_samples` | `None` | Stricter core point requirement when increased. None = same as min_cluster_size |
| `cluster_selection_epsilon` | `0.0` | Merges clusters closer than this threshold. Leave at 0.0 to disable |
| `cluster_selection_method` | `eom` | `eom` finds larger stable clusters. `leaf` finds smaller tighter ones |

Note: `metric` should stay as `euclidean` when `USE_UMAP = True`. If you disable UMAP and want cosine distance, set `metric="cosine"` and `algorithm="generic"`.

## Cluster lookup

Look up which cluster a paper belongs to and see all papers in that cluster. No API calls needed.

```bash
venv/Scripts/python.exe cluster_lookup.py 38844093    # specific paper
venv/Scripts/python.exe cluster_lookup.py             # summary of all clusters
```

## Predict cluster for new text

Given a new abstract or free text, predicts which existing cluster it belongs to using the fitted UMAP + HDBSCAN models. Embeds via Pinecone API, then runs `approximate_predict` against the saved density landscape.

```bash
# By PMID (uses stored abstract, shows actual vs predicted)
venv/Scripts/python.exe predict_cluster.py 38844093

# By free text
venv/Scripts/python.exe predict_cluster.py "schizophrenia dopamine receptor treatment"

# Demo mode
venv/Scripts/python.exe predict_cluster.py
```

Note: if you retune clustering params in `hdbscan_cluster.py`, re-run it to regenerate the `.pkl` model files before using `predict_cluster.py`.

## Hybrid search

Find related papers using both dense (semantic) and sparse (lexical) embeddings via the `pubmed-hybrid` Pinecone index. Predicts cluster via majority vote from nearest neighbors (alternative to `predict_cluster.py`).

```bash
# First time: create and populate the hybrid index
venv/Scripts/python.exe setup_hybrid_index.py

# Search by PMID (excludes self from results)
venv/Scripts/python.exe find_related_hybrid.py 38844093

# Search by free text
venv/Scripts/python.exe find_related_hybrid.py "heart failure treatment outcomes"
```

## Data

| File | Description |
|---|---|
| `data/articles.json` | 100 PubMed abstracts, 10 per medical category |
| `data/embeddings.json` | 1024-dim dense vectors from Pinecone multilingual-e5-large |
| `data/sparse_embeddings.json` | 100 sparse vectors from pinecone-sparse-english-v0 (SPLADE) |
| `data/hdbscan_results.json` | Latest dense cluster assignments and metrics |
| `data/hdbscan_sparse_results.json` | Sparse cluster assignments and metrics |
| `data/cluster_comparison.png` | 2x2 dense vs sparse visualization |

## Previous pipeline scripts

`prev_steps/` contains the Phase 1 scripts used to fetch articles, generate embeddings, and set up the Pinecone index. These do not need to be re-run unless you want to rebuild the dataset from scratch.

## Metrics reference

- **Silhouette**: -1 to 1. Higher is better. Measures how well-separated clusters are.
- **Davies-Bouldin**: Lower is better. Measures cluster compactness vs separation.
- **ARI**: -1 to 1. Compares cluster assignments against the known 10 medical categories. 1 = perfect recovery, 0 = random.
- **Noise %**: Points the algorithm rejected as outliers. Under 15% is healthy.
- **Membership confidence**: Per-point probability of belonging to its assigned cluster. Low values (under 0.5) flag ambiguous papers.

### WHY UMAP

The core problem is that HDBSCAN needs to estimate local density so it needs to be able to say "these points are packed tightly together, those are sparse." That   only works if distance is meaningful. In 1024 dimensions it isn't. This is the curse of dimensionality, as you add dimensions, the volume of the space grows so fast that all points spread out and become roughly equidistant from each other. The ratio between the closest and farthest point converges toward 1. There are no dense regions because everything  is equally far from everything else. I tested this by running HDBSCAN on the original 1024 dimension vector and found 0 clusters, it then labeled everything noise, tuning hyperparameters turned everything into 1 cluster or had random results with extremely poor metrics.

Why not PCA?

PCA finds the directions of maximum variance and projects onto those. It's a linear operation, it can only rotate and squash the data. The meaningful structure in text embeddings is not linear. The relationship between a cardiology paper and a nephrology paper isn't a straight line in embedding space, it's a curved manifold. PCA would smear across that curve and destroy the local neighbourhood relationships that HDBSCAN depends on.

Why UMAP?

UMAP doesn't projec it learns based on parameters. It builds a graph of who each point's nearest neighbours actually are in 1024 dimensions, then finds a low-dimensional layout that keeps those neighbourhood relationships as intact as possible. Points that were semantically close stay close. Points that were far stay far. The local density structure is preserved, just in a space where distance is meaningful again. (defined better through metrics given in hdbscan cluster file) -> n_neighbours tells what we consider meaningful cluster and we could still play more with the values to find the sweet spot I am assuming the hyperparameters in this test will look different from the ones on our huge corpus of pubmed papers.
