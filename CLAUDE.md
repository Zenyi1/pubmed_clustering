# PubMed Clustering — Project Guide

## Goal

Use the **Valyu API** to fetch 10 PubMed articles for each of **10 medical categories** (100 articles total), extract only their **abstracts**, generate **embeddings** from those abstracts, and upsert them into a **Pinecone vector database** for downstream clustering and similarity search.

---

## Tech Stack

- **Valyu API** — search and retrieve PubMed articles
- **Pinecone** — vector database + built-in inference API for embeddings (server-side, no local model)
- **Python + python-dotenv** — scripting and env management
- **venv** — already set up

## Environment Variables (`.env`)

```
PINECONE_API_KEY=...
VALYU_API_KEY=...
```

## Medical Categories (10)

1. Cardiology
2. Oncology
3. Neurology
4. Infectious Disease
5. Endocrinology
6. Gastroenterology
7. Pulmonology
8. Rheumatology
9. Nephrology
10. Psychiatry

---

## Step-by-Step Plan

> **Convention:** Complete each step fully, verify output, then STOP and report before moving to the next step.

---

### STEP 1 — Install Dependencies

Install required packages into the existing venv:

```
venv/Scripts/python.exe -m pip install requests python-dotenv valyu pinecone
```

**How to install into this venv on Windows (Git Bash):**
Use `venv/Scripts/python.exe -m pip install <package>` — do NOT use `pip` directly as it may target the wrong Python.

Installed and verified:
- `requests` 2.32.5 ✅
- `python-dotenv` 1.2.1 ✅
- `valyu` 2.6.0 ✅
- `pinecone` 8.1.0 ✅

No `sentence-transformers` needed — embeddings handled by Pinecone's inference API server-side.

- **STOP** — confirm install succeeded before proceeding. ✅ DONE

---

### STEP 2 — Explore Valyu API

Write a small test script (`test_valyu.py`) to:
- Load `VALYU_API_KEY` from `.env`
- Make a single test query to the Valyu API for one medical category (e.g. "cardiology")
- Print the raw response to understand the schema (fields available: title, abstract, PMID, etc.)

- **STOP** — show sample response, confirm `abstract` field is present and usable before proceeding.

---

### STEP 3 — Fetch PubMed Articles via Valyu

Write `fetch_articles.py`:
- Loop over all 10 medical categories
- For each category, query Valyu API for 10 PubMed articles
- Extract and store: `pmid`, `title`, `abstract`, `category`
- Save results to `data/articles.json`
- Log how many articles were fetched per category

- **STOP** — inspect `data/articles.json`, confirm 100 articles total (10 per category) with non-empty abstracts before proceeding.

---

### STEP 4 — Set Up Pinecone Index with Integrated Embedding

Write `setup_pinecone.py`:
- Load `PINECONE_API_KEY` from `.env`
- Create a Pinecone index named `pubmed-abstracts` if it does not already exist
  - Use Pinecone's integrated inference model: `multilingual-e5-large`
  - dimension: `1024` (matches `multilingual-e5-large`)
  - metric: `cosine`
  - spec: serverless (cloud: `aws`, region: `us-east-1`)
- Print index stats to confirm it is ready

No local model download needed — Pinecone embeds text server-side via `pinecone.inference.embed()` or directly via upsert with the inference-enabled index.

- **STOP** — confirm index exists and is ready before proceeding.

---

### STEP 5 — Embed Abstracts via Pinecone Inference API

Write `embed_abstracts.py`:
- Load `PINECONE_API_KEY` and `data/articles.json`
- Use `pc.inference.embed(model="multilingual-e5-large", inputs=[abstract_text], parameters={"input_type": "passage"})` for each abstract
- Each embedding is a 1024-dimensional float vector
- Save embeddings + metadata to `data/embeddings.json` (list of `{id, embedding, metadata}`)
  - metadata: `pmid`, `title`, `category`, `abstract` (truncated to 500 chars for Pinecone metadata limit)
- Embed in batches of 10 to stay within API rate limits

- **STOP** — confirm embeddings file created, spot-check one vector length (should be 1024) before proceeding.

---

### STEP 6 — Upsert Embeddings into Pinecone

Write `upsert_to_pinecone.py`:
- Load `PINECONE_API_KEY` and `data/embeddings.json`
- Connect to `pubmed-abstracts` index
- Upsert all 100 vectors in batches of 50
- Each vector: `{id: pmid, values: embedding, metadata: {...}}`
- Print final index stats (total vector count should be 100)

- **STOP** — confirm 100 vectors in index, show index stats before declaring done.

---

## File Layout (after all steps complete)

```
pubmed_clustering/
├── .env                    # API keys (gitignored)
├── CLAUDE.md               # This file
├── data/
│   ├── articles.json       # Raw fetched articles
│   └── embeddings.json     # Embedded abstracts + metadata
├── test_valyu.py           # Step 2: API exploration
├── fetch_articles.py       # Step 3: Fetch articles
├── embed_abstracts.py      # Step 4: Generate embeddings
├── setup_pinecone.py       # Step 5: Create Pinecone index
├── upsert_to_pinecone.py   # Step 6: Upsert to Pinecone
└── venv/
```

---

## Notes

- All API keys come from `.env` — never hardcode them.
- If Valyu returns fewer than 10 results for a category, log a warning and continue.
- Pinecone free tier supports 1 serverless index — reuse if it already exists.
- Do not skip steps or combine them — each has a verification checkpoint.
