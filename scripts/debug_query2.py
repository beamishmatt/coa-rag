"""Debug script with multiple query variations."""
from openai import OpenAI
from src.config import OPENAI_API_KEY, DEFAULT_MODEL as MODEL
from src.state import load_state

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()
vs_id = state["vector_store_id"]

queries = [
    "friction ridge impressions crime scene evidence",
    "fingerprints lifted processed Crime Scene Services",
    "latent print examination MSP Massachusetts State Police",
    "evidence collected 73 School Street crime scene"
]

for question in queries:
    print(f"\n{'='*60}")
    print(f"Query: {question}")
    print("="*60)
    
    resp = client.responses.create(
        model=MODEL,
        input=question,
        tools=[{"type": "file_search", "vector_store_ids": [vs_id]}],
    )
    
    print(f"\nResponse excerpt: {resp.output_text[:500]}...")
