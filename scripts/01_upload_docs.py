from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY
from src.state import load_state, save_state
from src.ingest import upload_files, attach_files_to_vector_store, wait_until_ready

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()
vs_id = state["vector_store_id"]

docs_dir = Path("data/docs")
file_ids = upload_files(client, docs_dir)
attach_files_to_vector_store(client, vs_id, file_ids)
wait_until_ready(client, vs_id)

state["file_ids"] = state.get("file_ids", []) + file_ids
save_state(state)

print("Uploaded and attached files:", len(file_ids))