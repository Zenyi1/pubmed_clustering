# PubMed Clustering — CLAUDE.md

## Project State

Phase 1 (data pipeline) is complete. 100 PubMed abstracts are embedded and live in Pinecone.
Phase 2 begins here — clustering, analysis, and search on top of that database.

---

## Pinecone Index

| Property | Value |
|---|---|
| Index name | `pubmed-abstracts` |
| Embedding model | `multilingual-e5-large` (Pinecone inference API) |
| Dimensions | 1024 |
| Metric | cosine |
| Spec | serverless, AWS us-east-1 |
| Vectors | 100 (10 per medical category) |

**Categories:** cardiology, oncology, neurology, infectious disease, endocrinology, gastroenterology, pulmonology, rheumatology, nephrology, psychiatry

---

## Querying the Index

```python
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pubmed-abstracts")

# Embed query text — always use input_type="query" for search
embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["your query here"],
    parameters={"input_type": "query"},
)

# Search — filter by category is optional
results = index.query(
    vector=embedding[0].values,
    top_k=5,
    include_metadata=True,
    filter={"category": {"$eq": "oncology"}},  # optional
)
```

**Metadata fields on every vector:**
`pmid`, `title`, `abstract` (first 500 chars), `category`, `url`, `relevance_score`

---

## Valyu API (if re-fetching data)

```python
from valyu import Valyu
client = Valyu(api_key=os.getenv("VALYU_API_KEY"))
response = client.search(
    query="...",
    search_type="all",                            # must be "all" with included_sources
    max_num_results=20,                           # hard API limit — 403 if exceeded
    included_sources=["pubmed.ncbi.nlm.nih.gov"],
    response_length="medium",
)
```

**Gotchas:**
- Only accept PMID URLs (`/\d+/`) — PMC URLs (`/PMC\d+`) have no abstract section
- Strip `?dopt=Abstract` query params before URL matching
- Abstract lives between `## Abstract\n\n` and the next `##` heading in `content`
- Some categories need multiple query variants to yield 10 valid PMID articles

---

## Environment

```
PINECONE_API_KEY=...
VALYU_API_KEY=...
```

Run scripts with: `venv/Scripts/python.exe <script>.py` (Windows — do not use bare `pip` or `python`)
Install packages with: `venv/Scripts/python.exe -m pip install <package>`
Always open JSON files with `encoding='utf-8'` (Windows default cp1252 breaks on medical text)

## Installed Packages

| Package | Version | Purpose |
|---|---|---|
| `pinecone` | 8.1.0 | Vector DB + inference API |
| `valyu` | 2.6.0 | PubMed search API |
| `scikit-learn` | 1.8.0 | DBSCAN, OPTICS, TF-IDF, metrics |
| `numpy` | 2.4.2 | Numerical ops |
| `scipy` | 1.17.1 | Sparse matrices |
| `torch` | 2.10.0 | Deep learning backend |
| `transformers` | 5.2.0 | HuggingFace models |
| `sentence-transformers` | 5.2.3 | Sentence embeddings |
| `python-dotenv` | 1.2.1 | .env loading |
| `requests` | 2.32.5 | HTTP |
| `pydantic` | 2.12.5 | Data validation |
| `joblib` | 1.5.3 | Parallelism |

| `hdbscan` | 0.8.41 | HDBSCAN clustering |
| `umap-learn` | 0.5.11 | Dimensionality reduction pre-clustering |
| `numba` | 0.64.0 | JIT backend for umap/hdbscan |
| `pynndescent` | 0.6.0 | Nearest-neighbour graphs (umap dep) |

---

## Phase 2 — Clustering & Related Work Lookup

> Same convention: complete each step, verify, STOP before moving on.

### STEP 7 — Install Phase 2 Dependencies
```
venv/Scripts/python.exe -m pip install hdbscan umap-learn
```
Needed: `hdbscan` (HDBSCAN algorithm), `umap-learn` (dimensionality reduction).

---

### STEP 8 — Dense Clustering (`cluster_dense.py`)
- Load 1024-dim vectors from `data/embeddings.json`
- Reduce to 15 dims with **UMAP** (metric=cosine) — required before DBSCAN/OPTICS
- Run all three algorithms:
  - **DBSCAN** — `eps` grid-searched via silhouette score, `min_samples=3`
  - **HDBSCAN** — `min_cluster_size=3`, `metric=euclidean` (post-UMAP space)
  - **OPTICS** — `min_samples=3`, produces reachability plot
- Metrics per algorithm: silhouette score, Davies-Bouldin index, ARI vs known 10 categories, noise %
- Save cluster assignments to `data/clusters_dense.json`
- Print side-by-side metrics comparison table

**Algorithm guide:**
- HDBSCAN: best overall — variable density, no epsilon, soft memberships
- DBSCAN: solid baseline — epsilon tuned via silhouette over UMAP space
- OPTICS: kept for reachability plot — shows density structure visually

---

### STEP 9 — Sparse Clustering (`cluster_sparse.py`)
- Build TF-IDF matrix from raw abstracts (`data/articles.json`)
  - `max_features=500`, `stop_words='english'`, `sublinear_tf=True`
- Run same three algorithms (DBSCAN, HDBSCAN, OPTICS) on sparse TF-IDF vectors
- Same metrics: silhouette, Davies-Bouldin, ARI, noise %
- Save to `data/clusters_sparse.json`
- Print metrics table (sparse) alongside dense table for comparison

---

### STEP 10 — Find Related Work: Dense Lookup (`find_related.py`)
- Given a PMID, fetch its stored vector from Pinecone (`index.fetch([pmid])`)
- Query top-k nearest neighbors (exclude self)
- Print ranked results with score, title, category, PMID
- Works entirely off the existing `pubmed-abstracts` index

---

### STEP 11 — Hybrid Pinecone Index (`setup_hybrid_index.py` + `upsert_hybrid.py`)
- Create new index `pubmed-hybrid`:
  - metric: `dotproduct` (required for sparse/hybrid in Pinecone)
  - dimension: 1024 (same dense model)
- Embed each abstract as **dense** (multilingual-e5-large, `input_type="passage"`)
- Embed each abstract as **sparse** (pinecone-sparse-english-v0)
- Upsert both into `pubmed-hybrid` index per vector

---

### STEP 12 — Find Related Work: Hybrid Lookup (`find_related_hybrid.py`)
- Given a PMID, embed its abstract as dense + sparse query vectors
- Query `pubmed-hybrid` with both (hybrid alpha blending: 0.5/0.5 to start)
- Compare results vs dense-only from Step 10
- Show where hybrid diverges — sparse helps surface shared terminology, dense handles semantics

---

## File Reference

| File | Purpose |
|---|---|
| `requirements.txt` | Direct dependencies — `pip install -r requirements.txt` to replicate |
| `data/articles.json` | 100 raw articles: pmid, title, abstract, category, url |
| `data/embeddings.json` | 100 vectors + metadata, ready to re-upsert if needed |
| `fetch_articles.py` | Re-fetch articles from Valyu (runs full pipeline) |
| `embed_abstracts.py` | Re-embed abstracts via Pinecone inference |
| `setup_pinecone.py` | Create index (idempotent — skips if exists) |
| `upsert_to_pinecone.py` | Upsert embeddings.json into Pinecone |
| `query_index.py` | Search the index by text query |
| `test_valyu.py` | Explore raw Valyu API response schema |
