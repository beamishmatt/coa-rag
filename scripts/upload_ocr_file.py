"""Upload the OCR'd text file to the vector store."""
from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY
from src.state import load_state, save_state
from src.ingest import wait_until_ready

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()
vs_id = state["vector_store_id"]

# Upload the OCR'd text file
txt_file = Path("data/docs/Crime Scene Services - MSP (10-18-2011).txt")
print(f"Uploading: {txt_file}")

f = client.files.create(file=open(txt_file, "rb"), purpose="assistants")
print(f"File ID: {f.id}")

# Attach to vector store
client.vector_stores.files.create(vector_store_id=vs_id, file_id=f.id)
print("Attached to vector store")

# Wait for indexing
print("Waiting for indexing...")
wait_until_ready(client, vs_id)
print("✅ Done! The OCR'd Crime Scene Services document is now searchable.")

# Save file ID to state
state["file_ids"] = state.get("file_ids", []) + [f.id]
save_state(state)




