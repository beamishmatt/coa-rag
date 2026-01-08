"""Debug script to see what chunks are retrieved for a query."""
from openai import OpenAI
from src.config import OPENAI_API_KEY, DEFAULT_MODEL as MODEL
from src.state import load_state

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()
vs_id = state["vector_store_id"]

question = "Were any latent prints lifted from the crime scene?"

print(f"Query: {question}\n")
print("="*60)

resp = client.responses.create(
    model=MODEL,
    input=question,
    tools=[{"type": "file_search", "vector_store_ids": [vs_id]}],
    include=["output[*].file_search_call.search_results"]
)

# Print search results
for output in resp.output:
    if hasattr(output, 'type') and output.type == 'file_search_call':
        print("\n📂 RETRIEVED CHUNKS:")
        if hasattr(output, 'search_results'):
            for i, result in enumerate(output.search_results):
                print(f"\n--- Chunk {i+1} (score: {result.score:.3f}) ---")
                print(f"File: {result.filename}")
                print(f"Content: {result.text[:500]}...")
        else:
            print("(No search_results attribute found)")

print("\n" + "="*60)
print("\n🤖 MODEL RESPONSE:")
print(resp.output_text)
