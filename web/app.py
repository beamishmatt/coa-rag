import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI
import uvicorn
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import src modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OPENAI_API_KEY, DEFAULT_MODEL
from src.state import load_state, save_state
from src.ingest import upload_files, attach_files_to_vector_store, wait_until_ready
from src.coa import coa_report_with_progress, stream_manager_response
from src.ask import ask_with_file_search
from web.websocket import InvestigationWebSocketManager

app = FastAPI(title="Investigative AI", description="ChatGPT-style investigative AI interface")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize WebSocket manager
manager = InvestigationWebSocketManager(client)

# Thread pool for running blocking operations
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_status():
    """Get system status and vector store information"""
    state = load_state()
    vector_store_id = state.get("vector_store_id")
    
    if not vector_store_id:
        return {"status": "not_initialized", "vector_store": None}
    
    try:
        vs = client.vector_stores.retrieve(vector_store_id)
        return {
            "status": "ready" if vs.file_counts.in_progress == 0 else "processing",
            "vector_store": {
                "id": vs.id,
                "name": vs.name,
                "file_counts": {
                    "in_progress": vs.file_counts.in_progress,
                    "completed": vs.file_counts.completed,
                    "total": vs.file_counts.total
                }
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/api/create-vector-store")
async def create_vector_store():
    """Create a new vector store"""
    
    if OPENAI_API_KEY == "sk-placeholder":
        raise HTTPException(
            status_code=400, 
            detail="Please add your OpenAI API key to the .env file to use this feature."
        )
    
    state = load_state()
    if state.get("vector_store_id"):
        raise HTTPException(status_code=400, detail="Vector store already exists")
    
    try:
        vs = client.vector_stores.create(name="investigative-ai-proto")
        state["vector_store_id"] = vs.id
        state["file_ids"] = []
        save_state(state)
        
        return {"vector_store_id": vs.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {str(e)}")

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents to the vector store"""
    
    if not OPENAI_API_KEY or OPENAI_API_KEY in ["sk-placeholder", "your_key_here"]:
        raise HTTPException(
            status_code=400, 
            detail="Please add your OpenAI API key to the .env file to enable document uploads."
        )
    
    state = load_state()
    vector_store_id = state.get("vector_store_id")
    
    if not vector_store_id:
        try:
            vs = client.vector_stores.create(name="investigative-ai-proto")
            vector_store_id = vs.id
            state["vector_store_id"] = vs.id
            state["file_ids"] = []
            save_state(state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create vector store: {str(e)}")
    
    try:
        file_ids = []
        for file in files:
            temp_path = Path(f"data/docs/{file.filename}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            with open(temp_path, "rb") as f:
                uploaded_file = client.files.create(file=f, purpose="assistants")
                file_ids.append(uploaded_file.id)
            
            client.vector_stores.files.create(
                vector_store_id=vector_store_id, 
                file_id=uploaded_file.id
            )
        
        state["file_ids"] = state.get("file_ids", []) + file_ids
        save_state(state)
        
        return {"uploaded_files": len(files), "file_ids": file_ids}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """List uploaded documents"""
    state = load_state()
    vector_store_id = state.get("vector_store_id")
    file_ids = state.get("file_ids", [])
    
    if not vector_store_id:
        return {"documents": []}
    
    documents = []
    for file_id in file_ids:
        try:
            file_info = client.files.retrieve(file_id)
            documents.append({
                "id": file_id,
                "filename": file_info.filename,
                "status": file_info.status,
                "created_at": file_info.created_at
            })
        except Exception:
            continue
    
    return {"documents": documents}

@app.delete("/api/documents/{file_id}")
async def delete_document(file_id: str):
    """Delete a document from the vector store"""
    state = load_state()
    
    try:
        client.files.delete(file_id)
        file_ids = state.get("file_ids", [])
        if file_id in file_ids:
            file_ids.remove(file_id)
            state["file_ids"] = file_ids
            save_state(state)
        
        return {"deleted": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "question":
                question = message_data["content"]
                history = message_data.get("history", [])
                await handle_streaming_question(websocket, question, history)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


async def handle_streaming_question(websocket: WebSocket, question: str, history: list = None):
    """Handle investigation questions with streaming response"""
    
    if history is None:
        history = []
    
    if OPENAI_API_KEY == "sk-placeholder":
        await manager.send_error(websocket, "Please add your OpenAI API key to the .env file.")
        return
    
    state = load_state()
    vector_store_id = state.get("vector_store_id")
    
    if not vector_store_id:
        await manager.send_error(websocket, "No documents uploaded. Please upload documents first.")
        return
    
    try:
        # Send initial stage
        await manager.send_stage_update(websocket, "workers", "Analyzing documents with worker agents...")
        
        # Capture the event loop BEFORE entering thread pool
        loop = asyncio.get_running_loop()
        
        # Track progress via queue
        progress_queue = asyncio.Queue()
        
        def on_worker_progress(status: str, current: int, total: int):
            """Callback to track worker progress (runs in thread)"""
            # Use the captured loop, not get_event_loop()
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(("worker", current, total, status)),
                loop
            )
        
        async def run_workers():
            return await loop.run_in_executor(
                executor,
                lambda: coa_report_with_progress(
                    client, DEFAULT_MODEL, vector_store_id, question, 
                    n_workers=4, on_progress=on_worker_progress,
                    conversation_history=history
                )
            )
        
        # Start workers and send progress updates
        worker_task = asyncio.create_task(run_workers())
        
        # Send progress updates while workers run
        workers_done = False
        while not workers_done:
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                if msg[0] == "worker":
                    _, current, total, status = msg
                    await manager.send_worker_progress(websocket, current, total, status)
            except asyncio.TimeoutError:
                pass
            
            if worker_task.done():
                workers_done = True
        
        # Get worker results
        worker_outputs, manager_input = await worker_task
        
        # Send synthesizing stage
        await manager.send_stage_update(websocket, "synthesizing", "Synthesizing findings...")
        await asyncio.sleep(0.3)  # Brief pause for UX
        
        # Signal stream start
        await manager.send_stream_start(websocket)
        
        # Stream the manager response
        def stream_response():
            """Generator wrapper for streaming (runs in thread)"""
            return list(stream_manager_response(client, DEFAULT_MODEL, manager_input))
        
        # Run streaming in thread and send chunks
        chunks = await loop.run_in_executor(executor, stream_response)
        
        for chunk in chunks:
            if chunk:
                await manager.send_chunk(websocket, chunk)
                await asyncio.sleep(0.01)  # Small delay for smooth rendering
        
        # Signal stream end
        await manager.send_stream_end(websocket, question)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        await manager.send_error(websocket, str(e))


@app.post("/api/ask")
async def ask_question(request: Dict[str, Any]):
    """Ask a question and get a response (non-streaming fallback)"""
    
    if OPENAI_API_KEY == "sk-placeholder":
        raise HTTPException(
            status_code=400, 
            detail="Please add your OpenAI API key to the .env file."
        )
    
    question = request.get("question", "")
    state = load_state()
    vector_store_id = state.get("vector_store_id")
    
    if not vector_store_id:
        raise HTTPException(status_code=400, detail="No vector store found")
    
    try:
        from src.coa import coa_report
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            executor,
            lambda: coa_report(client, DEFAULT_MODEL, vector_store_id, question, n_workers=4)
        )
        return {"response": report, "question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
