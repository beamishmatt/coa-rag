from pathlib import Path
from openai import OpenAI

def upload_files(client: OpenAI, docs_dir: Path) -> list[str]:
    file_ids = []
    for p in sorted(docs_dir.glob("*")):
        if not p.is_file():
            continue
        # Docs + cookbook examples commonly use purpose="assistants" for file_search flows.
        f = client.files.create(file=open(p, "rb"), purpose="assistants")
        file_ids.append(f.id)
    return file_ids

def attach_files_to_vector_store(client: OpenAI, vector_store_id: str, file_ids: list[str]) -> None:
    # simplest: attach individually (cookbook does this, parallelizable if you want)
    for fid in file_ids:
        client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=fid)

def wait_until_ready(client: OpenAI, vector_store_id: str, max_checks: int = 60) -> None:
    # The docs recommend waiting until file status is `completed` before querying.
    import time
    for _ in range(max_checks):
        vs = client.vector_stores.retrieve(vector_store_id)
        if vs.file_counts.in_progress == 0:
            return
        time.sleep(2)
    raise TimeoutError("Vector store still indexing after wait period")