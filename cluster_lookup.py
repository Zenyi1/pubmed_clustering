'''
Take local HDBSCAN results and lookup the corresponsing cluster based on PMID
'''

import json
import sys
from collections import defaultdict


with open("data/hdbscan_results.json", encoding="utf-8") as f:
    results = json.load(f)

assignments = results["assignments"] #they look like {"pmid": "38844093", "cluster": 12, "confidence": 1.0, "category": "cardiology", "title": "..."}

# Build lookup structures
by_pmid = {a["pmid"]: a for a in assignments}
by_cluster = defaultdict(list)
for a in assignments:
    by_cluster[a["cluster"]].append(a) #here we get {cluster_id: [list of assignment dicts]}


def show_paper(pmid):
    if pmid not in by_pmid:
        #paper is not in cluster
        print(f"PMID {pmid} not found in cluster data.")
        print(f"Available PMIDs: {', '.join(sorted(by_pmid.keys())[:10])}...")
        return

    paper = by_pmid[pmid]
    cid = paper["cluster"]

    print(f"Paper: {paper['title']}")
    print(f"  PMID:       {pmid}")
    print(f"  Category:   {paper['category']}")
    print(f"  Cluster:    {cid}{'  (NOISE)' if cid == -1 else ''}")
    print(f"  Confidence: {paper['confidence']}")

    if cid == -1:
        print("Noise paper, does not belong in any cluster")
        return

    cluster_papers = by_cluster[cid]
    others = [p for p in cluster_papers if p["pmid"] != pmid]
    others.sort(key=lambda x: -x["confidence"])

    #just cool visualizatoin of papers sorted by confidence of belinging to a cluster
    print(f"\n  Cluster {cid} has {len(cluster_papers)} papers total:")
    for p in others:
        print(f"    [{p['confidence']:.4f}] {p['category']:<22} {p['pmid']}  {p['title'][:70]}")


def show_summary():
    print(f"Clusters: {results['metrics']['n_clusters']}")
    print(f"Noise:    {results['metrics']['noise_pct']}%")
    print(f"ARI:      {results['metrics']['ari_vs_categories']}")
    print()

    for cid in sorted(by_cluster.keys()):
        papers = by_cluster[cid]
        label = "NOISE" if cid == -1 else f"Cluster {cid:>2}"
        cats = defaultdict(int)
        for p in papers:
            cats[p["category"]] += 1
        dominant = max(cats, key=cats.get)
        cat_str = ", ".join(f"{c}({n})" for c, n in sorted(cats.items(), key=lambda x: -x[1]))
        print(f"  {label}  ({len(papers):>2} papers)  {cat_str}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_paper(sys.argv[1])
    else:
        show_summary()
