import json
from pathlib import Path
from typing import Generator, Callable, Optional
from openai import OpenAI
from .ask import ask_with_file_search

def load_prompt(path: Path) -> str:
    return path.read_text()


def should_expand_query(question: str) -> bool:
    """
    Determine if a question would benefit from query expansion.
    Simple questions don't need expansion; complex ones do.
    """
    words = question.split()
    question_lower = question.lower()
    
    # Short questions usually don't need expansion
    if len(words) < 6:
        return False
    
    # Questions with multiple aspects benefit from expansion
    multi_aspect_indicators = [
        " and ", " or ", "including", "such as", "especially",
        "relationship", "connection", "between", "compare",
        "timeline", "sequence", "history", "background"
    ]
    
    if any(indicator in question_lower for indicator in multi_aspect_indicators):
        return True
    
    # Questions with multiple entities benefit from expansion
    # (rough heuristic: multiple capitalized words that aren't at sentence start)
    caps_words = [w for w in words[1:] if w and w[0].isupper()]
    if len(caps_words) >= 2:
        return True
    
    # Longer questions usually benefit
    if len(words) >= 12:
        return True
    
    return False


def decompose_query(
    client: OpenAI, 
    model: str, 
    question: str, 
    n_variants: int = 4,
    force_expand: bool = False
) -> tuple[list[str], bool]:
    """
    Generate diverse search queries from a user question.
    
    Returns: (list of search queries, whether expansion was used)
    """
    
    # Check if expansion is needed
    if not force_expand and not should_expand_query(question):
        return [question] * n_variants, False
    
    prompt = f"""Generate {n_variants} different search queries to find information for this investigation question.

RULES:
- Each query should target a DIFFERENT aspect, angle, or entity
- Use DIFFERENT vocabulary to maximize semantic search coverage
- Keep queries focused and specific
- Include variations that might surface edge cases or related context

QUESTION: {question}

Return ONLY a valid JSON array of exactly {n_variants} search query strings.
Example format: ["query about aspect 1", "query about aspect 2", "query about aspect 3", "query about aspect 4"]"""
    
    try:
        resp = client.responses.create(model=model, input=prompt)
        response_text = resp.output_text.strip()
        
        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)
        
        queries = json.loads(response_text)
        
        # Validate we got a list of strings
        if isinstance(queries, list) and len(queries) >= n_variants:
            queries = [str(q) for q in queries[:n_variants]]
            return queries, True
        else:
            # Not enough queries, pad with original
            queries = [str(q) for q in queries]
            while len(queries) < n_variants:
                queries.append(question)
            return queries[:n_variants], True
            
    except json.JSONDecodeError as e:
        print(f"Query decomposition JSON error: {e}")
        return [question] * n_variants, False
    except Exception as e:
        print(f"Query decomposition error: {e}")
        return [question] * n_variants, False

def coa_report(
    client: OpenAI,
    model: str,
    vector_store_id: str,
    question: str,
    n_workers: int = 4,
    use_query_expansion: bool = True,
) -> str:
    """Non-streaming version for backward compatibility"""
    worker_prompt = load_prompt(Path("prompts/worker.md"))
    manager_prompt = load_prompt(Path("prompts/manager.md"))

    # Query expansion: generate diverse search queries for workers
    if use_query_expansion:
        search_queries, expanded = decompose_query(client, model, question, n_workers)
    else:
        search_queries = [question] * n_workers
        expanded = False

    worker_outputs = []
    for i in range(n_workers):
        search_query = search_queries[i]
        
        worker_input = (
            f"{worker_prompt}\n\n"
            f"ORIGINAL USER QUESTION:\n{question}\n\n"
            f"YOUR SEARCH FOCUS:\n{search_query}\n\n"
            f"WORKER PASS: {i+1}/{n_workers}\n"
            f"Search for information related to your focus area while keeping the original question in mind."
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
    use_query_expansion: bool = True,
) -> tuple[list, str]:
    """
    Run CoA analysis with progress callbacks for workers.
    Returns (worker_outputs, manager_input) so streaming can happen separately.
    
    If use_query_expansion is True, generates diverse search queries for each worker
    to maximize retrieval coverage across different semantic spaces.
    """
    worker_prompt = load_prompt(Path("prompts/worker.md"))
    manager_prompt = load_prompt(Path("prompts/manager.md"))
    
    # Format conversation context
    history_context = format_conversation_history(conversation_history) if conversation_history else ""

    # Query expansion: generate diverse search queries for workers
    if use_query_expansion:
        if on_progress:
            on_progress("Analyzing question and generating search strategies...", 0, n_workers)
        search_queries, expanded = decompose_query(client, model, question, n_workers)
        if expanded:
            print(f"Query expanded into {n_workers} search variants")
    else:
        search_queries = [question] * n_workers
        expanded = False

    worker_outputs = []
    for i in range(n_workers):
        search_query = search_queries[i]
        
        if on_progress:
            # Show the search focus if expanded, otherwise generic message
            if expanded and search_query != question:
                status = f"Worker {i+1} searching: \"{search_query[:40]}{'...' if len(search_query) > 40 else ''}\""
            else:
                status = f"Worker {i+1} analyzing documents..."
            on_progress(status, i + 1, n_workers)
        
        worker_input = (
            f"{worker_prompt}\n\n"
            f"{history_context}"
            f"ORIGINAL USER QUESTION:\n{question}\n\n"
            f"YOUR SEARCH FOCUS:\n{search_query}\n\n"
            f"WORKER PASS: {i+1}/{n_workers}\n"
            f"Search for information related to your focus area while keeping the original question in mind."
        )
        r = ask_with_file_search(client, model, vector_store_id, worker_input, include_search_results=False)
        text = r.output_text
        try:
            parsed = json.loads(text)
            # Add metadata about which query this worker used
            parsed["_search_query"] = search_query
            worker_outputs.append(parsed)
        except Exception:
            worker_outputs.append({"raw_text": text, "_search_query": search_query})

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
