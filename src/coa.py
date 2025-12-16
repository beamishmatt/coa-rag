import json
from pathlib import Path
from typing import Generator, Callable, Optional
from openai import OpenAI
from .ask import ask_with_file_search

def load_prompt(path: Path) -> str:
    return path.read_text()

def coa_report(
    client: OpenAI,
    model: str,
    vector_store_id: str,
    question: str,
    n_workers: int = 4,
) -> str:
    """Non-streaming version for backward compatibility"""
    worker_prompt = load_prompt(Path("prompts/worker.md"))
    manager_prompt = load_prompt(Path("prompts/manager.md"))

    worker_outputs = []
    for i in range(n_workers):
        worker_input = (
            f"{worker_prompt}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"WORKER PASS: {i+1}/{n_workers}\n"
            f"Try to surface different angles, entities, dates, and edge cases."
        )
        r = ask_with_file_search(client, model, vector_store_id, worker_input, include_search_results=False)
        text = r.output_text
        try:
            worker_outputs.append(json.loads(text))
        except Exception:
            worker_outputs.append({"raw_text": text})

    manager_input = (
        f"{manager_prompt}\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"WORKER OUTPUTS (JSON):\n{json.dumps(worker_outputs, indent=2)}\n"
    )
    manager_resp = client.responses.create(model=model, input=manager_input)
    return manager_resp.output_text


def format_conversation_history(history: list) -> str:
    """Format conversation history for inclusion in prompts."""
    if not history:
        return ""
    
    formatted = "CONVERSATION HISTORY:\n"
    for msg in history:
        # Support both 'role' (from frontend) and 'sender' formats
        role_value = msg.get("role") or msg.get("sender", "")
        role = "User" if role_value in ["user", "User"] else "Assistant"
        content = msg.get("content", "")
        # Truncate long messages to avoid token limits
        if len(content) > 500:
            content = content[:500] + "..."
        formatted += f"{role}: {content}\n\n"
    
    return formatted + "---\n\n"


def coa_report_with_progress(
    client: OpenAI,
    model: str,
    vector_store_id: str,
    question: str,
    n_workers: int = 4,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    conversation_history: Optional[list] = None,
) -> tuple[list, str]:
    """
    Run CoA analysis with progress callbacks for workers.
    Returns (worker_outputs, manager_input) so streaming can happen separately.
    """
    worker_prompt = load_prompt(Path("prompts/worker.md"))
    manager_prompt = load_prompt(Path("prompts/manager.md"))
    
    # Format conversation context
    history_context = format_conversation_history(conversation_history) if conversation_history else ""

    worker_outputs = []
    for i in range(n_workers):
        if on_progress:
            on_progress(f"Worker {i+1} analyzing documents...", i + 1, n_workers)
        
        worker_input = (
            f"{worker_prompt}\n\n"
            f"{history_context}"
            f"CURRENT QUESTION:\n{question}\n\n"
            f"WORKER PASS: {i+1}/{n_workers}\n"
            f"Try to surface different angles, entities, dates, and edge cases."
        )
        r = ask_with_file_search(client, model, vector_store_id, worker_input, include_search_results=False)
        text = r.output_text
        try:
            worker_outputs.append(json.loads(text))
        except Exception:
            worker_outputs.append({"raw_text": text})

    manager_input = (
        f"{manager_prompt}\n\n"
        f"{history_context}"
        f"CURRENT QUESTION:\n{question}\n\n"
        f"WORKER OUTPUTS (JSON):\n{json.dumps(worker_outputs, indent=2)}\n"
    )
    
    return worker_outputs, manager_input


def stream_manager_response(
    client: OpenAI,
    model: str,
    manager_input: str,
) -> Generator[str, None, None]:
    """
    Stream the manager's response token by token.
    Falls back to yielding full response if streaming isn't supported.
    """
    try:
        # Try streaming first
        stream = client.responses.create(
            model=model,
            input=manager_input,
            stream=True
        )
        
        for event in stream:
            # Handle different event types from the Responses API
            if hasattr(event, 'type'):
                if event.type == 'response.output_text.delta':
                    if hasattr(event, 'delta'):
                        yield event.delta
                elif event.type == 'response.content_part.delta':
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        yield event.delta.text
                elif event.type == 'content_block_delta':
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        yield event.delta.text
            # Also check for direct text attribute
            elif hasattr(event, 'text'):
                yield event.text
            elif hasattr(event, 'delta'):
                if isinstance(event.delta, str):
                    yield event.delta
                elif hasattr(event.delta, 'text'):
                    yield event.delta.text
                    
    except Exception as e:
        # Fallback: If streaming fails, get full response and simulate streaming
        print(f"Streaming not available, falling back to full response: {e}")
        try:
            response = client.responses.create(model=model, input=manager_input)
            full_text = response.output_text
            
            # Simulate streaming by yielding chunks
            chunk_size = 10  # characters per chunk
            for i in range(0, len(full_text), chunk_size):
                yield full_text[i:i + chunk_size]
        except Exception as e2:
            yield f"Error generating response: {e2}"
