from openai import OpenAI
from src.config import OPENAI_API_KEY
from src.state import load_state
import sys

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()

vs_id = state.get("vector_store_id")
file_ids = state.get("file_ids", [])

# Vector store files can be removed from the store; file deletion is separate.
if vs_id:
    client.vector_stores.delete(vs_id)
    print("Deleted vector store:", vs_id)

for fid in file_ids:
    client.files.delete(fid)

print("Deleted files:", len(file_ids))
print("You may now delete .state.json manually.")