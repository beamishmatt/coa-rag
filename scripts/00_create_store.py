from openai import OpenAI
from src.config import OPENAI_API_KEY
from src.state import load_state, save_state

client = OpenAI(api_key=OPENAI_API_KEY)

state = load_state()
if state.get("vector_store_id"):
    print("Vector store already exists:", state["vector_store_id"])
    raise SystemExit(0)

# Vector store create API
vs = client.vector_stores.create(name="investigative-ai-proto")
state["vector_store_id"] = vs.id
state["file_ids"] = []
save_state(state)

print("Created vector store:", vs.id)