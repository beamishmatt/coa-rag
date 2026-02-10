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
from src.extract import (
    extract_from_document, load_extracted, save_extracted, 
    merge_extraction, detect_conflicts, get_extraction_summary,
    remove_document_extraction, deduplicate_extracted_data,
    analyze_investigative_notes
)
from src.router import (
    classify_query, answer_exhaustive_query, should_use_extracted_data, 
    is_guilt_query, get_guilt_deflection_response,
    is_speculative_query, get_speculative_deflection_response,
    get_query_category
)
from web.websocket import InvestigationWebSocketManager


def ocr_pdf(pdf_path: Path) -> str:
    """Extract text from scanned PDF using OCR"""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=300)
        
        # OCR each page
        text_parts = []
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            text_parts.append(f"--- Page {i+1} ---\n{page_text}")
        
        return "\n\n".join(text_parts)
    except ImportError as e:
        print(f"OCR dependencies not installed: {e}")
        return ""
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


def build_graph_data(extracted_data: dict, filter_names: List[str] = None) -> Dict[str, Any]:
    """
    Build graph data structure from extracted relationships for D3.js visualization.
    
    Args:
        extracted_data: The extracted data containing relationships and entities
        filter_names: Optional list of names to filter relationships (show only relationships involving these people)
    
    Returns:
        Dict with 'nodes' and 'links' arrays for D3.js force-directed graph
    """
    relationships = extracted_data.get("relationships", [])
    entities = extracted_data.get("entities", [])
    
    if not relationships:
        return {"nodes": [], "links": []}
    
    # Build entity lookup for descriptions
    entity_lookup = {}
    for entity in entities:
        if entity.get("type", "").lower() == "person":
            name_key = entity.get("name", "").lower().strip()
            entity_lookup[name_key] = entity
    
    # Collect unique nodes and build links
    nodes_dict = {}  # name_key -> node data
    links = []
    
    def normalize_name(name: str) -> str:
        return name.lower().strip()
    
    def get_entity_info(name: str) -> dict:
        """Look up entity info by name (fuzzy match)"""
        name_norm = normalize_name(name)
        
        # Exact match
        if name_norm in entity_lookup:
            return entity_lookup[name_norm]
        
        # Partial match (name parts)
        name_parts = set(name_norm.split())
        for key, entity in entity_lookup.items():
            key_parts = set(key.split())
            if name_parts & key_parts:  # Any common parts
                return entity
        
        return None
    
    # Filter relationships if names provided
    filtered_rels = relationships
    if filter_names:
        filter_names_norm = [normalize_name(n) for n in filter_names]
        filtered_rels = []
        for rel in relationships:
            p1_norm = normalize_name(rel.get("person1", ""))
            p2_norm = normalize_name(rel.get("person2", ""))
            p1_words = set(p1_norm.split())
            p2_words = set(p2_norm.split())
            
            # Check if either person matches any filter name
            for filter_name in filter_names_norm:
                filter_words = set(filter_name.split())
                if (filter_name in p1_norm or filter_name in p2_norm or
                    filter_words & p1_words or filter_words & p2_words):
                    filtered_rels.append(rel)
                    break
    
    for rel in filtered_rels:
        person1 = rel.get("person1", "Unknown")
        person2 = rel.get("person2", "Unknown")
        rel_type = rel.get("type", "unknown")
        description = rel.get("description", "")
        quote = rel.get("quote", "")
        source = rel.get("source", "")
        
        # Normalize keys
        p1_key = normalize_name(person1)
        p2_key = normalize_name(person2)
        
        # Add nodes if not already present
        if p1_key not in nodes_dict:
            entity_info = get_entity_info(person1)
            nodes_dict[p1_key] = {
                "id": p1_key,
                "name": person1,
                "type": "Person",
                "description": entity_info.get("description", "") if entity_info else "",
                "sources": list(set([source] if isinstance(source, str) else source)) if source else []
            }
        
        if p2_key not in nodes_dict:
            entity_info = get_entity_info(person2)
            nodes_dict[p2_key] = {
                "id": p2_key,
                "name": person2,
                "type": "Person",
                "description": entity_info.get("description", "") if entity_info else "",
                "sources": list(set([source] if isinstance(source, str) else source)) if source else []
            }
        
        # Add link
        links.append({
            "source": p1_key,
            "target": p2_key,
            "type": rel_type,
            "description": description,
            "quote": quote[:200] + "..." if len(quote) > 200 else quote,
            "source_doc": source if isinstance(source, str) else ", ".join(source) if source else ""
        })
    
    return {
        "nodes": list(nodes_dict.values()),
        "links": links
    }


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
    """Upload documents to the vector store and extract structured data"""
    
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
        extraction_results = []
        loop = asyncio.get_running_loop()
        
        for file in files:
            temp_path = Path(f"data/docs/{file.filename}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = await file.read()
            with open(temp_path, "wb") as buffer:
                buffer.write(content)
            
            # Upload to vector store
            with open(temp_path, "rb") as f:
                uploaded_file = client.files.create(file=f, purpose="assistants")
                file_ids.append(uploaded_file.id)
            
            client.vector_stores.files.create(
                vector_store_id=vector_store_id, 
                file_id=uploaded_file.id
            )
            
            # Extract structured data for preprocessing
            # For PDFs, we need to extract text first
            doc_text = ""
            if file.filename.lower().endswith('.pdf'):
                try:
                    import fitz  # PyMuPDF
                    pdf_doc = fitz.open(temp_path)
                    for page in pdf_doc:
                        doc_text += page.get_text()
                    pdf_doc.close()
                    
                    # If no text extracted, try OCR (scanned PDF)
                    if not doc_text.strip():
                        print(f"No text found in {file.filename}, attempting OCR...")
                        doc_text = await loop.run_in_executor(
                            executor,
                            lambda: ocr_pdf(temp_path)
                        )
                        if doc_text:
                            print(f"OCR extracted {len(doc_text)} characters from {file.filename}")
                        else:
                            print(f"OCR failed for {file.filename}")
                            
                except ImportError:
                    # PyMuPDF not installed, skip extraction
                    print(f"PyMuPDF not installed, skipping extraction for {file.filename}")
                    doc_text = ""
                except Exception as e:
                    print(f"PDF extraction error for {file.filename}: {e}")
                    doc_text = ""
            else:
                # Try to read as text
                try:
                    doc_text = content.decode('utf-8')
                except:
                    doc_text = ""
            
            # Run LLM extraction if we have text
            if doc_text:
                extraction = await loop.run_in_executor(
                    executor,
                    lambda: extract_from_document(client, DEFAULT_MODEL, doc_text, file.filename)
                )
                extraction_results.append({
                    "filename": file.filename,
                    "entities": len(extraction.get("entities", [])),
                    "claims": len(extraction.get("claims", [])),
                    "events": len(extraction.get("events", []))
                })
                
                # Merge with existing extractions
                all_data = load_extracted()
                all_data = merge_extraction(all_data, extraction, file.filename)
                
                # Detect conflicts across all documents
                all_data["conflicts"] = await loop.run_in_executor(
                    executor,
                    lambda: detect_conflicts(all_data, client, DEFAULT_MODEL)
                )
                save_extracted(all_data)
        
        state["file_ids"] = state.get("file_ids", []) + file_ids
        save_state(state)
        
        return {
            "uploaded_files": len(files), 
            "file_ids": file_ids,
            "extraction": extraction_results,
            "extraction_summary": get_extraction_summary()
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    """Delete a document from the vector store, OpenAI files, local storage, and extracted data"""
    state = load_state()
    vector_store_id = state.get("vector_store_id")
    
    try:
        # Get the filename before deleting (needed for local file and extraction removal)
        filename = None
        try:
            file_info = client.files.retrieve(file_id)
            filename = file_info.filename
        except Exception as e:
            print(f"Could not retrieve file info for {file_id}: {e}")
        
        # 1. Remove from vector store first (if vector store exists)
        if vector_store_id:
            try:
                client.vector_stores.files.delete(
                    vector_store_id=vector_store_id,
                    file_id=file_id
                )
            except Exception as e:
                print(f"Could not remove from vector store: {e}")
        
        # 2. Delete from OpenAI files
        try:
            client.files.delete(file_id)
        except Exception as e:
            print(f"Could not delete OpenAI file: {e}")
        
        # 3. Delete local file from data/docs/
        if filename:
            local_path = Path(f"data/docs/{filename}")
            if local_path.exists():
                try:
                    local_path.unlink()
                    print(f"Deleted local file: {local_path}")
                except Exception as e:
                    print(f"Could not delete local file: {e}")
            
            # 4. Remove extracted data for this document
            try:
                remove_document_extraction(filename)
                print(f"Removed extracted data for: {filename}")
            except Exception as e:
                print(f"Could not remove extracted data: {e}")
        
        # 5. Update state
        file_ids = state.get("file_ids", [])
        if file_id in file_ids:
            file_ids.remove(file_id)
            state["file_ids"] = file_ids
            save_state(state)
        
        return {"deleted": file_id, "filename": filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        # Check for guilt-related queries FIRST - before any processing
        if is_guilt_query(question):
            response = get_guilt_deflection_response()
            await manager.send_stream_start(websocket)
            
            # Send deflection response in chunks for consistent UX
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                await manager.send_chunk(websocket, response[i:i + chunk_size])
                await asyncio.sleep(0.01)
            
            await manager.send_stream_end(websocket, question)
            return
        
        # Check for speculative queries (motive, who benefits, why killed)
        if is_speculative_query(question):
            response = get_speculative_deflection_response()
            await manager.send_stream_start(websocket)
            
            # Send deflection response in chunks for consistent UX
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                await manager.send_chunk(websocket, response[i:i + chunk_size])
                await asyncio.sleep(0.01)
            
            await manager.send_stream_end(websocket, question)
            return
        
        loop = asyncio.get_running_loop()
        
        # Load extracted data for entity-aware routing
        extracted_data = await loop.run_in_executor(executor, load_extracted)
        
        # Route query to appropriate handler (now entity-aware)
        query_type = classify_query(question, extracted_data)
        print(f"\n{'='*60}")
        print(f"ROUTING DEBUG: Question: {question[:80]}...")
        print(f"ROUTING DEBUG: Query type: {query_type}")
        print(f"{'='*60}\n")
        
        if query_type == "EXHAUSTIVE":
            # Use preprocessed extracted data for exhaustive queries
            # Send "graph" stage to trigger simplified loading UI
            await manager.send_stage_update(websocket, "graph", "Querying knowledge graph...")
            
            response, success = await loop.run_in_executor(
                executor,
                lambda: answer_exhaustive_query(question, extracted_data, client, DEFAULT_MODEL)
            )
            
            # Check if this is a relationship query - if so, include graph data
            query_category = get_query_category(question, extracted_data)
            graph_data = None
            
            if query_category == "relationships":
                # Build graph data for visualization
                graph_data = build_graph_data(extracted_data)
                print(f"GRAPH DEBUG: Built graph with {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('links', []))} links")
            
            # Now that response is ready, signal stream start
            await manager.send_stream_start(websocket)
            
            # Send response in chunks for consistent UX
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                await manager.send_chunk(websocket, response[i:i + chunk_size])
                await asyncio.sleep(0.01)
            
            # Signal stream end with optional graph data
            await manager.send_stream_end(websocket, question, graph_data)
            return
        
        # SPECIFIC queries use CoA + file_search
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
        
        # Send synthesizing stage - keep this visible while LLM processes
        await manager.send_stage_update(websocket, "synthesizing", "Synthesizing findings...")
        
        # Stream the manager response
        def stream_response():
            """Generator wrapper for streaming (runs in thread)"""
            return list(stream_manager_response(client, DEFAULT_MODEL, manager_input))
        
        # Run streaming in thread and send chunks
        # Keep "synthesizing" loading state visible until response is ready
        chunks = await loop.run_in_executor(executor, stream_response)
        
        # Now that we have content ready, signal stream start (removes loading indicator)
        await manager.send_stream_start(websocket)
        
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
        # Check for guilt-related queries FIRST
        if is_guilt_query(question):
            return {
                "response": get_guilt_deflection_response(), 
                "question": question, 
                "query_type": "deflected"
            }
        
        # Check for speculative queries (motive, who benefits, why killed)
        if is_speculative_query(question):
            return {
                "response": get_speculative_deflection_response(), 
                "question": question, 
                "query_type": "deflected"
            }
        
        loop = asyncio.get_running_loop()
        
        # Load extracted data for entity-aware routing
        extracted_data = await loop.run_in_executor(executor, load_extracted)
        
        # Route query to appropriate handler (now entity-aware)
        query_type = classify_query(question, extracted_data)
        
        if query_type == "EXHAUSTIVE":
            response, success = await loop.run_in_executor(
                executor,
                lambda: answer_exhaustive_query(question, extracted_data, client, DEFAULT_MODEL)
            )
            return {"response": response, "question": question, "query_type": "exhaustive"}
        
        # SPECIFIC queries use CoA
        from src.coa import coa_report
        report = await loop.run_in_executor(
            executor,
            lambda: coa_report(client, DEFAULT_MODEL, vector_store_id, question, n_workers=4)
        )
        return {"response": report, "question": question, "query_type": "specific"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/extraction")
async def get_extraction_status():
    """Get status of preprocessed extracted data"""
    try:
        extracted_data = load_extracted()
        summary = get_extraction_summary(extracted_data)
        return {
            "status": "ready" if summary["documents"] > 0 else "empty",
            "summary": summary,
            "documents": extracted_data.get("documents", []),
            "conflicts_count": len(extracted_data.get("conflicts", []))
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/extraction/conflicts")
async def get_conflicts():
    """Get all detected conflicts/inconsistencies"""
    try:
        extracted_data = load_extracted()
        return {
            "conflicts": extracted_data.get("conflicts", []),
            "total": len(extracted_data.get("conflicts", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/extraction/entities")
async def get_entities():
    """Get all extracted entities"""
    try:
        extracted_data = load_extracted()
        return {
            "entities": extracted_data.get("entities", []),
            "total": len(extracted_data.get("entities", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extraction/deduplicate")
async def deduplicate_extraction():
    """Deduplicate entities in the extracted data"""
    try:
        extracted_data = load_extracted()
        before_count = len(extracted_data.get("entities", []))
        
        extracted_data = deduplicate_extracted_data(extracted_data)
        save_extracted(extracted_data)
        
        after_count = len(extracted_data.get("entities", []))
        
        return {
            "status": "success",
            "entities_before": before_count,
            "entities_after": after_count,
            "duplicates_removed": before_count - after_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reextract/{filename:path}")
async def reextract_document(filename: str):
    """Re-extract a document using OCR (for scanned PDFs)"""
    
    if not OPENAI_API_KEY or OPENAI_API_KEY in ["sk-placeholder", "your_key_here"]:
        raise HTTPException(status_code=400, detail="API key not configured")
    
    # Find the file in data/docs
    file_path = Path(f"data/docs/{filename}")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    try:
        loop = asyncio.get_running_loop()
        
        # Try regular text extraction first
        doc_text = ""
        if filename.lower().endswith('.pdf'):
            import fitz
            pdf_doc = fitz.open(file_path)
            for page in pdf_doc:
                doc_text += page.get_text()
            pdf_doc.close()
            
            # If no text, use OCR
            if not doc_text.strip():
                print(f"Using OCR for {filename}...")
                doc_text = await loop.run_in_executor(executor, lambda: ocr_pdf(file_path))
        
        if not doc_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from document")
        
        # Remove old extraction for this document
        remove_document_extraction(filename)
        
        # Run new extraction
        extraction = await loop.run_in_executor(
            executor,
            lambda: extract_from_document(client, DEFAULT_MODEL, doc_text, filename)
        )
        
        # Merge with existing data
        all_data = load_extracted()
        all_data = merge_extraction(all_data, extraction, filename)
        
        # Re-detect conflicts
        all_data["conflicts"] = await loop.run_in_executor(
            executor,
            lambda: detect_conflicts(all_data, client, DEFAULT_MODEL)
        )
        save_extracted(all_data)
        
        return {
            "filename": filename,
            "extracted": {
                "entities": len(extraction.get("entities", [])),
                "claims": len(extraction.get("claims", [])),
                "events": len(extraction.get("events", [])),
                "key_facts": len(extraction.get("key_facts", []))
            },
            "text_length": len(doc_text),
            "summary": get_extraction_summary()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Re-extraction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
