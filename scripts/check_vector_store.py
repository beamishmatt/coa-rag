"""Check vector store file status."""
from openai import OpenAI
from src.config import OPENAI_API_KEY
from src.state import load_state

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()
vs_id = state["vector_store_id"]

print(f"Vector Store ID: {vs_id}")
print("\nFiles in vector store:")
print("-" * 60)

vs_files = client.vector_stores.files.list(vector_store_id=vs_id)
for f in vs_files.data:
    # Get file details
    file_info = client.files.retrieve(f.id)
    print(f"\n📄 {file_info.filename}")
    print(f"   ID: {f.id}")
    print(f"   Status: {f.status}")
    print(f"   Size: {file_info.bytes:,} bytes")
    if hasattr(f, 'last_error') and f.last_error:
        print(f"   ⚠️ ERROR: {f.last_error}")
