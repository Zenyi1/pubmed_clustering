import json
from dotenv import load_dotenv
import os
from valyu import Valyu

load_dotenv()

client = Valyu(api_key=os.getenv("VALYU_API_KEY"))

response = client.search(
    query="cardiology heart failure treatment",
    search_type="all",
    max_num_results=5,
    included_sources=["pubmed.ncbi.nlm.nih.gov"], #should only check from these sources
    response_length="medium",
)


#abstract included in content
if response and response.results:
    r = response.results[0]
    print("All fields on result[0]:")
    print(json.dumps(r.model_dump(), indent=2, ensure_ascii=True, default=str))
else:
    print("No results or error.")
    print(response)