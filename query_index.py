import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "pubmed-abstracts"
EMBEDDING_MODEL = "multilingual-e5-large"
TOP_K = 5

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)


def query(text: str, top_k: int = TOP_K, filter_category: str = None):
    """Embed a query string and return the top_k most similar abstracts."""
    embedding = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[text],
        parameters={"input_type": "query"},  # "query" for search, "passage" for documents
    )

    filter_dict = {"category": {"$eq": filter_category}} if filter_category else None

    results = index.query(
        vector=embedding[0].values,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
    )

    return results.matches


def print_results(matches, query_text):
    print(f'\nQuery: "{query_text}"')
    print("-" * 60)
    for i, match in enumerate(matches, 1):
        m = match.metadata
        print(f"{i}. [{m['category'].upper()}] score={match.score:.4f}")
        print(f"   Title:    {m['title'][:80]}")
        print(f"   PMID:     {m['pmid']}  |  {m['url']}")
        print(f"   Abstract: {m['abstract'][:150]}...")
        print()


if __name__ == "__main__":
    # Test 1: broad medical query
    matches = query("inflammation and immune response in chronic disease")
    print_results(matches, "inflammation and immune response in chronic disease")

    # Test 2: specific clinical query
    matches = query("antidepressant efficacy in major depressive disorder")
    print_results(matches, "antidepressant efficacy in major depressive disorder")

    # Test 3: filtered to one category
    matches = query("treatment outcomes and survival rates", filter_category="oncology")
    print_results(matches, "treatment outcomes and survival rates [oncology only]")
