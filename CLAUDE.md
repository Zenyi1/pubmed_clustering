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
Always open JSON files with `encoding='utf-8'` (Windows default cp1252 breaks on medical text)

---

## File Reference

| File | Purpose |
|---|---|
| `data/articles.json` | 100 raw articles: pmid, title, abstract, category, url |
| `data/embeddings.json` | 100 vectors + metadata, ready to re-upsert if needed |
| `fetch_articles.py` | Re-fetch articles from Valyu (runs full pipeline) |
| `embed_abstracts.py` | Re-embed abstracts via Pinecone inference |
| `setup_pinecone.py` | Create index (idempotent — skips if exists) |
| `upsert_to_pinecone.py` | Upsert embeddings.json into Pinecone |
| `query_index.py` | Search the index by text query |
| `test_valyu.py` | Explore raw Valyu API response schema |
