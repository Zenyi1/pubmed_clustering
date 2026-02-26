import json
import re
import os
import time
from dotenv import load_dotenv
from valyu import Valyu

load_dotenv()

client = Valyu(api_key=os.getenv("VALYU_API_KEY"))

#two distinct queries per category to maximise PMID-bearing results
CATEGORIES = {
    "cardiology":        ["cardiology heart failure treatment outcomes",
                          "cardiac arrhythmia myocardial infarction therapy"],
    "oncology":          ["oncology cancer chemotherapy clinical trial",
                          "tumor immunotherapy targeted therapy outcomes"],
    "neurology":         ["neurology stroke brain clinical study",
                          "epilepsy multiple sclerosis neurological treatment"],
    "infectious disease":["infectious disease antibiotic resistance treatment",
                          "COVID-19 antiviral treatment clinical outcomes",
                          "HIV antiretroviral therapy adherence outcomes",
                          "sepsis bacterial pneumonia antimicrobial management",
                          "tuberculosis malaria tropical disease treatment"],
    "endocrinology":     ["endocrinology diabetes mellitus insulin therapy",
                          "thyroid disorder hormonal treatment clinical trial",
                          "obesity metabolic syndrome glucagon-like peptide",
                          "adrenal cortisol pituitary endocrine disorder"],
    "gastroenterology":  ["gastroenterology inflammatory bowel disease treatment",
                          "liver cirrhosis colorectal cancer clinical outcomes",
                          "Crohn disease ulcerative colitis biologic therapy"],
    "pulmonology":       ["pulmonology COPD asthma respiratory treatment",
                          "lung disease pneumonia pulmonary clinical study",
                          "pulmonary fibrosis sleep apnea respiratory outcomes"],
    "rheumatology":      ["rheumatology rheumatoid arthritis treatment outcomes",
                          "lupus autoimmune disease clinical therapy",
                          "gout ankylosing spondylitis biologic DMARD treatment"],
    "nephrology":        ["nephrology chronic kidney disease dialysis treatment",
                          "renal failure glomerulonephritis clinical outcomes"],
    "psychiatry":        ["psychiatry depression antidepressant clinical trial",
                          "anxiety disorder cognitive behavioral therapy outcomes",
                          "schizophrenia bipolar disorder psychopharmacology",
                          "PTSD ADHD mental health treatment randomized trial"],
}

#filter direct PMID URLs: pubmed.ncbi.nlm.nih.gov/<digits>/
PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?")


def extract_abstract(content, description):
    # Primary: look for ## Abstract heading
    match = re.search(r"## Abstract\n\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: description field (truncated abstract snippet from Valyu)
    if description and len(description.strip()) > 80:
        return description.strip()
    return None


def fetch_category(category, queries, target=10):
    collected = {}  # pmid -> article dict, dedup within category

    for query in queries:
        if len(collected) >= target:
            break

        response = client.search(
            query=query,
            search_type="all",
            max_num_results=20,  #max api limit
            included_sources=["pubmed.ncbi.nlm.nih.gov"],
            response_length="medium",
        )

        
        if not response or not response.results:
            #print(f"No results for query: '{query}'")
            continue
        

        for r in response.results:
            if len(collected) >= target:
                break

            # Guard 1: PMID-only URLs (not PMC — those lack abstracts)
            url_clean = r.url.split("?")[0]
            pmid_match = PMID_URL_RE.search(url_clean) #regex claude made above
            if not pmid_match:
                continue

            pmid = pmid_match.group(1)
            if pmid in collected:
                continue

            content = str(r.content) if r.content else ""

            # Guard 2: must have a parseable abstract
            abstract = extract_abstract(content, r.description or "")
            if not abstract:
                print(f"  [SKIP] No abstract: PMID {pmid}")
                continue

            title = r.title.removesuffix(" - PubMed").strip()

            collected[pmid] = {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "category": category,
                "url": r.url.split("?")[0],
                "relevance_score": r.relevance_score,
            }

        time.sleep(0.5)  # brief pause between queries

    return list(collected.values())


def main():
    all_articles = []
    global_seen_pmids = set()

    for category, queries in CATEGORIES.items():
        print(f"Fetching: {category} ...")
        articles = fetch_category(category, queries)

        # Global dedup across categories
        unique = []
        for a in articles:
            if a["pmid"] not in global_seen_pmids:
                global_seen_pmids.add(a["pmid"])
                unique.append(a)
            else:
                print(f"  [DEDUP] PMID {a['pmid']} already in another category")

        count = len(unique)
        print(f"  -> {count}/10 valid articles", "" if count == 10 else "[WARN: fewer than 10]")
        all_articles.extend(unique)
        time.sleep(1)

    os.makedirs("data", exist_ok=True)
    with open("data/articles.json", "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_articles)} total articles saved to data/articles.json")
    by_cat = {}
    for a in all_articles:
        by_cat.setdefault(a["category"], 0)
        by_cat[a["category"]] += 1
    for cat, count in by_cat.items():
        status = "OK" if count >= 10 else f"WARN: only {count}"
        print(f"  {cat}: {count} [{status}]")


if __name__ == "__main__":
    main()
