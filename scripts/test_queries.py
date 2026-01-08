"""Test queries against the newly uploaded docs."""
from openai import OpenAI
from src.config import OPENAI_API_KEY, DEFAULT_MODEL as MODEL
from src.state import load_state

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()
vs_id = state["vector_store_id"]

queries = [
    "Were any latent prints lifted from the crime scene?",
    "What did Seth Green say in his statement?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print("="*60)
    
    resp = client.responses.create(
        model=MODEL,
        input=q,
        tools=[{"type": "file_search", "vector_store_ids": [vs_id]}],
    )
    print(f"\nA: {resp.output_text[:800]}...")
