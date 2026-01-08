"""Try very specific queries that should only hit the CSI report."""
from openai import OpenAI
from src.config import OPENAI_API_KEY, DEFAULT_MODEL as MODEL
from src.state import load_state

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()
vs_id = state["vector_store_id"]

# These details are from the extracted.json - they came from the CSI PDF
queries = [
    "Trooper Todd R. Girouard report",
    "Sergeant Kerry A. Gilpin approved",  
    "Seth Green fingerprint left middle finger",
    "friction ridge impression 2-9 individualized",
]

for question in queries:
    print(f"\n{'='*60}")
    print(f"Query: {question}")
    
    resp = client.responses.create(
        model=MODEL,
        input=f"Find this exact information: {question}",
        tools=[{"type": "file_search", "vector_store_ids": [vs_id]}],
    )
    
    # Check if the response mentions the CSI report
    has_csi = "Crime Scene" in resp.output_text or "Girouard" in resp.output_text or "Gilpin" in resp.output_text
    print(f"Found CSI content: {'✅ YES' if has_csi else '❌ NO'}")
    print(f"Response: {resp.output_text[:300]}...")
