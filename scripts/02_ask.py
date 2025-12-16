import sys
from openai import OpenAI
from src.config import OPENAI_API_KEY, DEFAULT_MODEL
from src.state import load_state
from src.ask import ask_with_file_search

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()

question = " ".join(sys.argv[1:]).strip() or "Give me a timeline and key findings."
resp = ask_with_file_search(
    client,
    DEFAULT_MODEL,
    state["vector_store_id"],
    question,
    include_search_results=True,  # cookbook pattern
)

print(resp.output_text)

# Optional: print raw search results if present
# (Structure can vary; include is the supported way to expose them)