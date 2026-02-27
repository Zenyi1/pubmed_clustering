"""
Manual dataset with a ground truth knowledge of these clusters
"""

import json
import re
import os
import time
from dotenv import load_dotenv
from valyu import Valyu
from pinecone import Pinecone

load_dotenv()


CLUSTERS = {
    0: {
        "name": "CAR-T cell therapy for B-cell lymphoma",
        "target": 35,
        "queries": [
            "CAR-T cell therapy diffuse large B-cell lymphoma outcomes",
            "chimeric antigen receptor T cell CD19 lymphoma clinical trial",
            "axicabtagene ciloleucel tisagenlecleucel B-cell lymphoma",
            "CAR-T cytokine release syndrome B-cell lymphoma management",
            "CAR-T cell therapy relapsed refractory B-cell lymphoma",
            "CD19 CAR-T complete remission lymphoma long-term",
            "CAR-T cell expansion persistence B-cell malignancy",
            "lisocabtagene maraleucel brexucabtagene B-cell lymphoma",
        ],
    },
    1: {
        "name": "GLP-1 receptor agonists for type 2 diabetes",
        "target": 30,
        "queries": [
            "GLP-1 receptor agonist type 2 diabetes glycemic control",
            "semaglutide liraglutide type 2 diabetes clinical trial",
            "incretin therapy type 2 diabetes HbA1c reduction",
            "GLP-1 agonist cardiovascular outcomes type 2 diabetes",
            "semaglutide weight loss type 2 diabetes metabolic",
            "dulaglutide exenatide type 2 diabetes comparison",
            "GLP-1 receptor agonist beta cell function insulin",
        ],
    },
    2: {
        "name": "Deep brain stimulation for Parkinson's disease",
        "target": 20,
        "queries": [
            "deep brain stimulation Parkinson disease motor outcomes",
            "subthalamic nucleus stimulation Parkinson clinical trial",
            "DBS globus pallidus Parkinson tremor rigidity",
            "deep brain stimulation Parkinson quality of life",
            "adaptive deep brain stimulation Parkinson closed-loop",
            "DBS Parkinson long-term follow-up motor fluctuations",
        ],
    },
    3: {
        "name": "Immune checkpoint inhibitors (PD-1/PD-L1) for NSCLC",
        "target": 15,
        "queries": [
            "PD-1 PD-L1 inhibitor non-small cell lung cancer outcomes",
            "pembrolizumab nivolumab NSCLC clinical trial",
            "immune checkpoint inhibitor NSCLC first-line therapy",
            "atezolizumab durvalumab non-small cell lung cancer",
            "PD-L1 expression biomarker NSCLC immunotherapy response",
        ],
    },
}

EMBEDDING_MODEL = "multilingual-e5-large"
BATCH_SIZE = 10
ABSTRACT_METADATA_LIMIT = 500

PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?")

# ---------------------------------------------------------------------------
# Helpers (reused from prev_steps/fetch_articles.py)
# ---------------------------------------------------------------------------


def extract_abstract(content, description):
    """Parse abstract from Valyu content, fall back to description."""
    match = re.search(r"## Abstract\n\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    if description and len(description.strip()) > 80:
        return description.strip()
    return None


def fetch_cluster(client, cluster_id, config, global_seen):
    """Fetch articles for one cluster, deduplicating against global_seen."""
    collected = {}
    name = config["name"]
    target = config["target"]

    for query in config["queries"]:
        if len(collected) >= target:
            break

        response = client.search(
            query=query,
            search_type="all",
            max_num_results=20,
            included_sources=["pubmed.ncbi.nlm.nih.gov"],
            response_length="medium",
        )

        if not response or not response.results:
            continue

        for r in response.results:
            if len(collected) >= target:
                break

            url_clean = r.url.split("?")[0]
            pmid_match = PMID_URL_RE.search(url_clean)
            if not pmid_match:
                continue

            pmid = pmid_match.group(1)
            if pmid in collected or pmid in global_seen:
                continue

            content = str(r.content) if r.content else ""
            abstract = extract_abstract(content, r.description or "")
            if not abstract:
                print(f"  [SKIP] No abstract: PMID {pmid}")
                continue

            title = r.title.removesuffix(" - PubMed").strip()

            collected[pmid] = {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "cluster_id": cluster_id,
                "cluster_name": name,
                "url": url_clean,
                "relevance_score": r.relevance_score,
            }

        time.sleep(0.5)

    return list(collected.values())


# ---------------------------------------------------------------------------
# Embedding (reused from prev_steps/embed_abstracts.py)
# ---------------------------------------------------------------------------


def embed_articles(pc, articles):
    """Embed all articles using Pinecone inference API."""
    embeddings = []

    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i : i + BATCH_SIZE]
        abstracts = [a["abstract"] for a in batch]

        result = pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=abstracts,
            parameters={"input_type": "passage"},
        )

        for article, embedding in zip(batch, result):
            embeddings.append({
                "id": article["pmid"],
                "values": embedding.values,
                "metadata": {
                    "pmid": article["pmid"],
                    "title": article["title"],
                    "abstract": article["abstract"][:ABSTRACT_METADATA_LIMIT],
                    "cluster_id": article["cluster_id"],
                    "cluster_name": article["cluster_name"],
                    "url": article["url"],
                    "relevance_score": article["relevance_score"],
                },
            })

        print(f"  Embedded {min(i + BATCH_SIZE, len(articles))}/{len(articles)}")
        time.sleep(0.3)

    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    client = Valyu(api_key=os.getenv("VALYU_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    all_articles = []
    global_seen = set()

    print("=== Fetching articles ===\n")

    for cluster_id, config in CLUSTERS.items():
        name = config["name"]
        target = config["target"]
        print(f"Cluster {cluster_id}: {name} (target: {target})")

        articles = fetch_cluster(client, cluster_id, config, global_seen)

        for a in articles:
            global_seen.add(a["pmid"])

        count = len(articles)
        status = "OK" if count >= target else f"SHORT by {target - count}"
        print(f"  -> {count}/{target} [{status}]\n")

        all_articles.extend(articles)
        time.sleep(1)

    # Save articles
    os.makedirs("data", exist_ok=True)
    with open("data/known_clusters.json", "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_articles)} articles to data/known_clusters.json")

    # Embed
    print("\n=== Embedding articles ===\n")
    embeddings = embed_articles(pc, all_articles)

    with open("data/known_clusters_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)
    print(f"Saved {len(embeddings)} embeddings to data/known_clusters_embeddings.json")

    # Summary
    print("\n=== Summary ===\n")
    counts = {}
    for a in all_articles:
        key = f"Cluster {a['cluster_id']}: {a['cluster_name']}"
        counts[key] = counts.get(key, 0) + 1

    for key, count in sorted(counts.items()):
        target = CLUSTERS[int(key.split(":")[0].split()[-1])]["target"]
        status = "OK" if count >= target else f"SHORT by {target - count}"
        print(f"  {key}: {count}/{target} [{status}]")

    print(f"\n  Total unique PMIDs: {len(all_articles)}")

    # Sanity check: print 2 titles per cluster (ASCII-safe for Windows)
    print("\n=== Sample titles ===\n")
    for cluster_id, config in CLUSTERS.items():
        print(f"Cluster {cluster_id} ({config['name']}):")
        cluster_articles = [a for a in all_articles if a["cluster_id"] == cluster_id]
        for a in cluster_articles[:2]:
            safe_title = a["title"].encode("ascii", errors="replace").decode("ascii")
            print(f"  - {safe_title}")
        print()

    # Verify embedding dimensions
    if embeddings:
        dim = len(embeddings[0]["values"])
        print(f"Embedding dimension: {dim}")
        assert dim == 1024, f"Expected 1024, got {dim}"
        print("Dimension check: OK")


if __name__ == "__main__":
    main()
