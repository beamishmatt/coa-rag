"""
Upload documents to vector store.
Automatically OCRs scanned PDFs before uploading.

Usage:
  python scripts/01_upload_docs.py          # Upload new docs only
  python scripts/01_upload_docs.py --fresh  # Clear and re-upload all
"""
import sys
from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY
from src.state import load_state, save_state
from src.ingest import upload_files, attach_files_to_vector_store, wait_until_ready

client = OpenAI(api_key=OPENAI_API_KEY)
state = load_state()
vs_id = state["vector_store_id"]

# Check for --fresh flag to clear existing files
if "--fresh" in sys.argv:
    print("🗑️  Clearing existing files from vector store...")
    existing_files = client.vector_stores.files.list(vector_store_id=vs_id)
    for f in existing_files.data:
        try:
            client.vector_stores.files.delete(vector_store_id=vs_id, file_id=f.id)
            client.files.delete(f.id)
            print(f"   Deleted: {f.id}")
        except Exception as e:
            print(f"   Warning: Could not delete {f.id}: {e}")
    state["file_ids"] = []
    print("✅ Cleared existing files\n")

# Also clean up any .txt files that were previously OCR'd (we'll regenerate them)
if "--fresh" in sys.argv:
    docs_dir = Path("data/docs")
    for txt_file in docs_dir.glob("*.txt"):
        pdf_counterpart = docs_dir / f"{txt_file.stem}.pdf"
        if pdf_counterpart.exists():
            print(f"🗑️  Removing old OCR file: {txt_file.name}")
            txt_file.unlink()

print("="*60)
print("📤 UPLOADING DOCUMENTS (with auto-OCR for scanned PDFs)")
print("="*60)

docs_dir = Path("data/docs")
file_ids = upload_files(client, docs_dir, auto_ocr=True)

print("\n" + "="*60)
print("🔗 Attaching files to vector store...")
attach_files_to_vector_store(client, vs_id, file_ids)

print("⏳ Waiting for indexing...")
wait_until_ready(client, vs_id)

state["file_ids"] = state.get("file_ids", []) + file_ids
save_state(state)

print("\n" + "="*60)
print(f"✅ SUCCESS! Uploaded {len(file_ids)} files")
print("="*60)