# PubMed Clustering — CLAUDE.md

## Project State

Phase 1 complete — 100 PubMed abstracts fetched, embedded, and stored in Pinecone.
Phase 2 complete — sparse clustering, hybrid index, related work lookup, and cluster prediction all done.
Phase 3 in progress — curated ground-truth dataset with 4 known clusters, HDBSCAN achieves perfect ARI=1.0.

---

## Clustering Findings (Phase 2)

**Winner: HDBSCAN + UMAP (cosine -> 40 dims euclidean)**

Comparison across all setups tested:

| Setup | Clusters | Noise% | ARI |
|---|---|---|---|
| No UMAP + cosine (generic) | 0 | 100% | 0.00 |
| No UMAP + euclidean | 2 | 33% | 0.05 |
| UMAP(15d) + HDBSCAN euclidean | 12 | 7% | 0.60 |
| UMAP(15d) + DBSCAN | 7 | 72% | 0.04 |
| UMAP(15d) + OPTICS | 16 | 18% | 0.49 |

UMAP is required — raw 1024-dim vectors cause all density-based methods to collapse (curse of dimensionality).
Use `euclidean` in HDBSCAN post-UMAP. Cosine in HDBSCAN requires `algorithm="generic"` but produced 100% noise on raw vectors.
DBSCAN excluded from further work — too sensitive to epsilon, 72% noise even with UMAP.
OPTICS excluded — viable but HDBSCAN dominates on ARI and noise%.

**Active tuning script:** `hdbscan_cluster.py`
Current params: `n_components=40`, `n_neighbors=5`, `min_cluster_size=3`, `method=eom`

**Step 9 result — Dense vs Sparse (pinecone-sparse-english-v0 + UMAP + HDBSCAN):**

| Metric | Dense | Sparse |
|---|---|---|
| Clusters | 13 | 11 |
| Noise % | 1.0% | 4.0% |
| Silhouette | 0.8276 | 0.7475 |
| Davies-Bouldin | 0.2331 | 0.2953 |
| ARI | 0.6287 | 0.4583 |
| Avg confidence | 0.8655 | 0.8884 |

Dense wins on all quality metrics. Sparse ARI is 0.46 vs 0.63 — confirms semantic embeddings capture category structure better than lexical/SPLADE weights. Sparse merges semantically related categories (e.g. pulmonology + infectious disease) because surface vocabulary overlaps. Dense keeps them separate via learned semantics.

Pinecone sparse embedding API note: `emb.sparse_indices` (list of large int hash IDs) and `emb.sparse_values` (list of floats). Hashes are NOT bounded vocabulary indices — must remap to compact indices before building `scipy.sparse.csr_matrix`. 4057 unique tokens across 100 docs, avg 150.7 non-zero per abstract. UMAP accepts sparse matrices natively with `metric="cosine"`.

Visualization: `visualize_clusters.py` generates `data/cluster_comparison.png` — 2x2 grid (dense/sparse x clusters/categories), 2D UMAP with `min_dist=0.1`.

---

## Phase 3 — Known-Cluster Dataset

**Goal:** Ground-truth dataset where cluster assignments are known in advance, so clustering quality can be properly evaluated. 4 hyper-specific research niches (not broad categories like Phase 2).

**Cluster design (imbalanced, 100 total):**

| ID | Topic | Papers |
|---|---|---|
| 0 | CAR-T cell therapy for B-cell lymphoma | 35 |
| 1 | GLP-1 receptor agonists for type 2 diabetes | 30 |
| 2 | Deep brain stimulation for Parkinson's disease | 20 |
| 3 | Immune checkpoint inhibitors (PD-1/PD-L1) for NSCLC | 15 |

**HDBSCAN results (same params as Phase 2: UMAP 40d + HDBSCAN eom):**

| Metric | Value |
|---|---|
| Clusters | 4 |
| Noise % | 0.0% |
| Silhouette | 0.9423 |
| Davies-Bouldin | 0.0875 |
| ARI vs ground truth | **1.0** (perfect) |
| Avg confidence | 0.9358 |

Perfect recovery — every paper assigned to its correct cluster with zero noise. Confirms that tight research niches + dense embeddings + UMAP+HDBSCAN pipeline = ideal clustering.

---

## Pinecone Indexes

### Dense-only: `pubmed-abstracts`

| Property | Value |
|---|---|
| Index name | `pubmed-abstracts` |
| Embedding model | `multilingual-e5-large` (Pinecone inference API) |
| Dimensions | 1024 |
| Metric | cosine |
| Spec | serverless, AWS us-east-1 |
| Vectors | 100 (10 per medical category) |

### Hybrid: `pubmed-hybrid`

| Property | Value |
|---|---|
| Index name | `pubmed-hybrid` |
| Dense model | `multilingual-e5-large` (1024d) |
| Sparse model | `pinecone-sparse-english-v0` (SPLADE) |
| Metric | dotproduct (required for hybrid) |
| Spec | serverless, AWS us-east-1 |
| Vectors | 100 (dense + sparse per vector) |

**Categories:** cardiology, oncology, neurology, infectious disease, endocrinology, gastroenterology, pulmonology, rheumatology, nephrology, psychiatry

Querying dense-only:
```python
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pubmed-abstracts")
embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["query text"],
    parameters={"input_type": "query"},
)
results = index.query(vector=embedding[0].values, top_k=5, include_metadata=True)
```

Querying hybrid:
```python
index = pc.Index("pubmed-hybrid")
dense = pc.inference.embed(model="multilingual-e5-large", inputs=["query"], parameters={"input_type": "query"})
sparse = pc.inference.embed(model="pinecone-sparse-english-v0", inputs=["query"], parameters={"input_type": "query"})
results = index.query(
    vector=dense[0].values,
    sparse_vector={"indices": list(sparse[0].sparse_indices), "values": list(sparse[0].sparse_values)},
    top_k=5, include_metadata=True,
)
```

Metadata per vector: `pmid`, `title`, `abstract` (500 chars), `category`, `url`, `relevance_score`

---

## Valyu API (if re-fetching)

```python
client = Valyu(api_key=os.getenv("VALYU_API_KEY"))
response = client.search(
    query="...",
    search_type="all",                            # must be "all" with included_sources
    max_num_results=20,                           # hard limit — 403 if exceeded
    included_sources=["pubmed.ncbi.nlm.nih.gov"],
    response_length="medium",
)
```

Gotchas: only accept PMID URLs (`/\d+/`), strip `?dopt=Abstract`, abstract is between `## Abstract\n\n` and next `##` heading, some categories need 3-5 query variants to fill 10 articles.

---

## Environment

```
PINECONE_API_KEY=...
VALYU_API_KEY=...
```

- Run: `venv/Scripts/python.exe <script>.py`
- Install: `venv/Scripts/python.exe -m pip install <package>`
- Always open JSON with `encoding='utf-8'` — Windows cp1252 breaks on medical text
- Avoid printing Unicode chars (e.g. block chars) to terminal — use ASCII fallbacks

## Installed Packages

| Package | Version | Purpose |
|---|---|---|
| `pinecone` | 8.1.0 | Vector DB + inference API |
| `valyu` | 2.6.0 | PubMed search API |
| `scikit-learn` | 1.8.0 | Metrics (silhouette, ARI, Davies-Bouldin) |
| `numpy` | 2.4.2 | Array ops |
| `scipy` | 1.17.1 | Sparse matrices |
| `hdbscan` | 0.8.41 | HDBSCAN clustering |
| `umap-learn` | 0.5.11 | Dimensionality reduction |
| `numba` | 0.64.0 | JIT backend for umap/hdbscan |
| `pynndescent` | 0.6.0 | Nearest-neighbour graphs (umap dep) |
| `python-dotenv` | 1.2.1 | .env loading |
| `requests` | 2.32.5 | HTTP |
| `torch` | 2.10.0 | Deep learning backend (transitive) |
| `transformers` | 5.2.0 | HuggingFace models (transitive) |
| `sentence-transformers` | 5.2.3 | Sentence embeddings (transitive) |
| `matplotlib` | 3.10.8 | Cluster comparison visualization |

---

## Phase 2 — Completed Steps

### STEP 9 — Sparse Clustering (`cluster_sparse.py`) — DONE
- Used pinecone-sparse-english-v0 (SPLADE) instead of TF-IDF — same HDBSCAN+UMAP pipeline
- Dense wins: ARI 0.63 vs 0.46, Silhouette 0.83 vs 0.75
- Visualization: `visualize_clusters.py` -> `data/cluster_comparison.png`

### STEPS 10-11 — Hybrid Index + Related Work Lookup — DONE
- Step 10 (dense-only KNN) removed — redundant since clusters already group related papers
- `setup_hybrid_index.py`: creates `pubmed-hybrid` index (dotproduct), upserts 100 vectors with dense+sparse embeddings
- `find_related_hybrid.py`: queries hybrid index by PMID or free text, predicts cluster via majority vote from top-5 neighbors
- `cluster_lookup.py`: pure local JSON lookup — shows cluster membership and co-clustered papers, no API calls

### Cluster Prediction (`predict_cluster.py`) — DONE
- Uses saved UMAP reducer + HDBSCAN clusterer (pickle files from `hdbscan_cluster.py`)
- Embeds new text via Pinecone dense model, transforms through fitted UMAP, runs `hdbscan.approximate_predict`
- Returns predicted cluster + confidence score directly from the density landscape (no neighbor vote)
- Must re-run `hdbscan_cluster.py` after retuning params to regenerate the .pkl files

---

## Code Style Preferences

- Don't separate code sections with big comment banners (e.g. `# -----------`). Use blank lines and concise inline comments instead.
- Console output (print statements) is fine, but when explaining scripts to the user, clearly distinguish what's **core logic** (UMAP, HDBSCAN, metrics, saving results) vs **visualization/display** (plots, per-cluster breakdowns, sample titles). Lead with what matters.

---

## File Reference

### Phase 1–2

| File | Purpose |
|---|---|
| `README.md` | Setup and usage guide |
| `requirements.txt` | Direct dependencies |
| `hdbscan_cluster.py` | Main clustering script — edit params at top to tune, saves .pkl models |
| `data/articles.json` | 100 raw articles: pmid, title, abstract, category, url |
| `data/embeddings.json` | 100 dense vectors + metadata |
| `data/hdbscan_results.json` | Latest cluster assignments and metrics |
| `data/clusters_dense.json` | DBSCAN/HDBSCAN/OPTICS comparison run (archived) |
| `data/sparse_embeddings.json` | 100 sparse vectors (pinecone-sparse-english-v0), cached |
| `data/hdbscan_sparse_results.json` | Sparse cluster assignments and metrics |
| `data/cluster_comparison.png` | 2x2 dense vs sparse visualization |
| `cluster_sparse.py` | Sparse clustering script (mirrors hdbscan_cluster.py) |
| `prev_steps/visualize_clusters.py` | 2x2 UMAP plot — dense vs sparse, clusters vs categories |
| `setup_hybrid_index.py` | Creates `pubmed-hybrid` index and upserts dense+sparse vectors |
| `find_related_hybrid.py` | Hybrid search by PMID or text, predicts cluster via neighbor vote |
| `cluster_lookup.py` | Local cluster lookup — shows cluster members for a PMID |
| `predict_cluster.py` | Predict cluster for new text via HDBSCAN approximate_predict |
| `data/umap_reducer.pkl` | Fitted UMAP model (regenerated by hdbscan_cluster.py) |
| `data/hdbscan_clusterer.pkl` | Fitted HDBSCAN model (regenerated by hdbscan_cluster.py) |
| `prev_steps/` | Phase 1 scripts: fetch, embed, index setup, upsert, query |

### Phase 3 — Known-Cluster Dataset

| File | Purpose |
|---|---|
| `build_manual_dataset.py` | Fetches 100 papers (Valyu) + embeds (Pinecone) for 4 ground-truth clusters |
| `cluster_known.py` | HDBSCAN+UMAP on the known-cluster dataset, saves results + .pkl models + 2D viz |
| `data/known_clusters.json` | 100 articles with `cluster_id` (0-3) and `cluster_name` |
| `data/known_clusters_embeddings.json` | Dense embeddings (1024-dim) with cluster metadata |
| `data/known_clusters_hdbscan_results.json` | Cluster assignments, metrics (ARI=1.0) |
| `data/known_clusters_umap_reducer.pkl` | Fitted UMAP for known-cluster data |
| `data/known_clusters_hdbscan_clusterer.pkl` | Fitted HDBSCAN for known-cluster data |
| `data/known_clusters_viz.png` | Side-by-side: HDBSCAN clusters vs ground truth (2D UMAP) |
