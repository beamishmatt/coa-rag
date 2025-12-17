"""
Query router module for directing queries to the appropriate handler.

Routing strategy:
- GRAPH: Entity lookups, comprehensive lists, conflicts, timelines - use preprocessed knowledge graph
- VECTOR: Deep analysis, complex reasoning, multi-hop questions - use CoA + file_search
- HYBRID: Start with graph, augment with vector if needed (future)
"""

import re
from typing import Tuple, Optional, List
from openai import OpenAI
from .extract import load_extracted, get_extraction_summary


def _synthesize_response(
    client: OpenAI,
    model: str,
    question: str,
    raw_data: str,
    category: str
) -> str:
    """
    Use LLM to synthesize a natural, well-organized response from extracted data.
    
    Args:
        client: OpenAI client
        model: Model to use
        question: Original user question
        raw_data: Template-formatted extracted data
        category: Query category (conflicts, entities, events, summary, general)
    
    Returns:
        Synthesized natural language response
    """
    
    system_prompt = """You are an investigative analyst assistant. Your job is to take structured extracted data and synthesize it into a clear, professional response that directly answers the user's question.

MARKDOWN FORMATTING RULES (CRITICAL):
- Use ## for main section headers (with blank line after)
- Use ### for subsection headers (with blank line after)  
- Use #### for sub-subsection headers when needed (with blank line after)
- NEVER use ##### or ###### - limit to 4 header levels max
- Use **bold** for emphasis on key names, dates, and important facts
- Use *italics* for sources and citations
- Use > for blockquotes when including direct quotes
- Use - for bullet lists (with blank line before the list)
- Use --- for horizontal rules to separate major sections
- ALWAYS include a blank line after headers and between paragraphs
- ALWAYS include a blank line before and after lists
- ALWAYS include a blank line before and after blockquotes
- Use single line breaks between items in lists, not double

CONTENT GUIDELINES:
- Write in a professional, analytical tone appropriate for investigative work
- Organize information logically based on what the user asked
- Highlight the most important findings first
- If there are conflicts or inconsistencies, explain their significance
- Always cite sources when available
- Be concise but thorough - don't pad with unnecessary text
- If the data doesn't fully answer the question, acknowledge limitations

CRITICAL ANTI-HALLUCINATION RULES:
- ONLY include information that appears in the extracted data provided
- If someone asks about a person/entity NOT in the data, say "No information found about [name] in the documents"
- NEVER invent names, dates, facts, or relationships not explicitly in the data
- If the extracted data is empty or doesn't contain relevant information, say so clearly
- Do not fill gaps with assumptions or general knowledge
- When uncertain, state "The documents do not specify..." rather than guessing"""

    user_prompt = f"""User's Question: {question}

Query Category: {category}

Extracted Data:
{raw_data}

Based on the extracted data above, provide a well-organized response using proper markdown formatting. Synthesize the information into a coherent narrative that directly answers the user's question."""

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return resp.output_text.strip()
    except Exception as e:
        print(f"LLM synthesis error: {e}")
        # Fall back to raw data if synthesis fails
        return raw_data


def _normalize_for_matching(text: str) -> str:
    """Normalize text for entity matching."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def _extract_potential_names(question: str) -> List[str]:
    """
    Extract potential entity names from a question.
    Looks for capitalized words (proper nouns) and quoted strings.
    """
    names = []
    
    # Extract quoted strings
    quoted = re.findall(r'["\']([^"\']+)["\']', question)
    names.extend(quoted)
    
    # Extract capitalized word sequences (proper nouns)
    # Match sequences like "John Smith", "Detective Roman", "Amanda Lynn Plasse"
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
    names.extend(capitalized)
    
    return names


def _find_matching_entities(names: List[str], entities: List[dict]) -> List[dict]:
    """
    Find entities that match any of the given names.
    Uses fuzzy matching to handle partial names.
    """
    matches = []
    
    for name in names:
        name_normalized = _normalize_for_matching(name)
        name_words = set(name_normalized.split())
        
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_normalized = _normalize_for_matching(entity_name)
            entity_words = set(entity_normalized.split())
            
            # Exact match
            if name_normalized == entity_normalized:
                if entity not in matches:
                    matches.append(entity)
                continue
            
            # Partial match - all words in query name appear in entity name
            if name_words and name_words.issubset(entity_words):
                if entity not in matches:
                    matches.append(entity)
                continue
            
            # Partial match - entity name words appear in query
            if entity_words and entity_words.issubset(name_words):
                if entity not in matches:
                    matches.append(entity)
                continue
            
            # Single word match for single-word queries (first or last name)
            if len(name_words) == 1 and len(entity_words) > 1:
                if name_words & entity_words:  # Any overlap
                    if entity not in matches:
                        matches.append(entity)
    
    return matches


def _is_entity_lookup_query(question: str) -> bool:
    """
    Determine if a question is primarily an entity lookup.
    These should go to the knowledge graph, not vector search.
    """
    question_lower = question.lower().strip()
    
    # Patterns that indicate entity lookup
    entity_lookup_patterns = [
        r'^who is\b',
        r'^who was\b', 
        r'^who are\b',
        r'^what is (?:the )?\w+(?:\'s| of)\b',  # "what is John's role"
        r'^tell me about\b',
        r'^what do (?:we|you) know about\b',
        r'^information (?:on|about)\b',
        r'^details (?:on|about)\b',
        r'^background on\b',
        r'^profile of\b',
        r'^describe\b',
        r'\bwho\b.*\bmentioned\b',
        r'\bwhat\b.*\brole\b',
    ]
    
    for pattern in entity_lookup_patterns:
        if re.search(pattern, question_lower):
            return True
    
    return False


def _is_comprehensive_query(question: str) -> bool:
    """
    Determine if a question requires comprehensive/exhaustive data.
    These should go to the knowledge graph.
    """
    question_lower = question.lower().strip()
    
    comprehensive_keywords = [
        "all ", "every ", "list ", "find all", "show all", "give me all",
        "inconsistencies", "contradictions", "conflicts", "discrepancies",
        "everyone", "everything", "everybody",
        "summarize all", "summary of all", "summarize the",
        "how many", "count ",
        "complete list", "full list",
        "all people", "all entities", "all events",
        "timeline", "chronology", "sequence of events",
        "overview", "what do we know",
        "what entities", "what people", "what events",
        "list the ", "list all",
    ]
    
    for keyword in comprehensive_keywords:
        if keyword in question_lower:
            return True
    
    return False


def _is_deep_analysis_query(question: str) -> bool:
    """
    Determine if a question requires deep analysis (vector + CoA).
    These need to search the actual documents for nuanced answers.
    """
    question_lower = question.lower().strip()
    
    # Patterns indicating need for deep document analysis
    deep_analysis_patterns = [
        r'why did\b',
        r'why was\b',
        r'how did\b',
        r'what happened\b.*\bwhen\b',
        r'what.*\bsay about\b',
        r'what.*\btestif',
        r'what.*\bstate\b',
        r'what.*\bclaim\b',
        r'explain.*\brelationship\b',
        r'connection between\b',
        r'evidence\b.*\b(?:that|of|for)\b',
        r'prove\b',
        r'according to\b',
        r'what does.*\b(?:document|report|interview)\b.*\bsay\b',
        r'quote\b',
        r'exact\b.*\bword',
        r'specific.*\bdetail',
        r'context\b.*\bof\b',
        r'circumstances\b',
        r'motive\b',
        r'reason\b.*\bfor\b',
    ]
    
    for pattern in deep_analysis_patterns:
        if re.search(pattern, question_lower):
            return True
    
    return False


def classify_query(question: str, extracted_data: dict = None) -> str:
    """
    Classify a query to determine optimal routing.
    
    Returns:
        "EXHAUSTIVE" - Use knowledge graph (preprocessed extracted data)
        "SPECIFIC" - Use vector search with CoA
    
    Routing logic:
    1. Comprehensive/list queries → EXHAUSTIVE (knowledge graph)
    2. Entity lookup queries → Check if entity exists in graph
       - Entity found → EXHAUSTIVE
       - Entity not found → SPECIFIC (search documents)
    3. Deep analysis queries → SPECIFIC (need document context)
    4. Default → SPECIFIC (CoA handles uncertainty well)
    """
    question_lower = question.lower().strip()
    
    # 1. Comprehensive queries always go to knowledge graph
    if _is_comprehensive_query(question):
        return "EXHAUSTIVE"
    
    # 2. Check for entity lookup patterns
    if _is_entity_lookup_query(question):
        # Load extracted data if not provided
        if extracted_data is None:
            extracted_data = load_extracted()
        
        entities = extracted_data.get("entities", [])
        
        if entities:
            # Extract potential entity names from question
            potential_names = _extract_potential_names(question)
            
            if potential_names:
                # Check if any mentioned entities exist in our graph
                matching_entities = _find_matching_entities(potential_names, entities)
                
                if matching_entities:
                    # Entity found in knowledge graph - use it
                    print(f"Router: Found {len(matching_entities)} matching entities in graph for: {potential_names}")
                    return "EXHAUSTIVE"
                else:
                    # Entity not in graph - need to search documents
                    print(f"Router: No matching entities found for: {potential_names}, using vector search")
                    return "SPECIFIC"
            else:
                # Generic entity query without specific name - use graph
                return "EXHAUSTIVE"
    
    # 3. Deep analysis queries need vector search
    if _is_deep_analysis_query(question):
        return "SPECIFIC"
    
    # 4. Default to SPECIFIC - CoA handles uncertainty well
    return "SPECIFIC"


def get_query_category(question: str, extracted_data: dict = None) -> str:
    """
    Get more detailed category for exhaustive queries to determine response type.
    
    Returns: "conflicts", "entities", "events", "summary", or "general"
    """
    question_lower = question.lower()
    
    if any(kw in question_lower for kw in ["inconsisten", "contradict", "conflict", "discrepan"]):
        return "conflicts"
    
    # Check for entity-related queries
    entity_keywords = ["people", "person", "everyone", "who", "entities", "organizations", 
                       "names", "name", "individuals", "suspects", "witnesses", "victims"]
    entity_patterns = ["tell me about", "information on", "details on", "background on", 
                       "profile of", "what do we know about", "describe"]
    
    if any(kw in question_lower for kw in entity_keywords):
        return "entities"
    
    if any(pattern in question_lower for pattern in entity_patterns):
        return "entities"
    
    # Check if question mentions a known entity name
    if extracted_data is None:
        extracted_data = load_extracted()
    
    potential_names = _extract_potential_names(question)
    if potential_names:
        entities = extracted_data.get("entities", [])
        if _find_matching_entities(potential_names, entities):
            return "entities"
    
    if any(kw in question_lower for kw in ["timeline", "events", "when", "chronolog", "sequence", "dates"]):
        return "events"
    
    if any(kw in question_lower for kw in ["summarize", "summary", "overview", "everything"]):
        return "summary"
    
    return "general"


def answer_exhaustive_query(
    question: str,
    extracted_data: dict = None,
    client: OpenAI = None,
    model: str = None
) -> Tuple[str, bool]:
    """
    Answer a query using preprocessed extracted data.
    
    If client and model are provided, uses LLM to synthesize a natural response.
    Otherwise, returns template-formatted data.
    
    Returns: (response_text, success)
    """
    if extracted_data is None:
        extracted_data = load_extracted()
    
    # Check if we have any data
    summary = get_extraction_summary(extracted_data)
    if summary["documents"] == 0:
        return ("No documents have been processed yet. Please upload documents first.", False)
    
    category = get_query_category(question, extracted_data)
    
    # Generate template-based response
    if category == "conflicts":
        raw_response, success = _answer_conflicts_query(extracted_data)
    elif category == "entities":
        raw_response, success = _answer_entities_query(question, extracted_data)
    elif category == "events":
        raw_response, success = _answer_events_query(extracted_data)
    elif category == "summary":
        raw_response, success = _answer_summary_query(extracted_data)
    else:
        raw_response, success = _answer_general_exhaustive(question, extracted_data)
    
    # If client provided, synthesize a natural response
    if client and model and success:
        synthesized = _synthesize_response(client, model, question, raw_response, category)
        return (synthesized, True)
    
    return (raw_response, success)


def _answer_conflicts_query(data: dict) -> Tuple[str, bool]:
    """Generate response for conflict/inconsistency queries."""
    
    conflicts = data.get("conflicts", [])
    
    if not conflicts:
        # Check if we have claims to analyze
        claims = data.get("claims", [])
        if not claims:
            return ("No claims were extracted from the documents to analyze for inconsistencies.", True)
        
        return (
            "## No Inconsistencies Detected\n\n"
            f"Analyzed {len(claims)} claims across {len(data.get('documents', []))} document(s). "
            "No direct contradictions or inconsistencies were identified.\n\n"
            "_Note: This is based on extracted claims. For deeper analysis, try asking specific questions about particular topics._",
            True
        )
    
    response = "## Detected Inconsistencies & Conflicts\n\n"
    response += f"Found **{len(conflicts)}** potential inconsistencies across the documents:\n\n"
    
    for i, conflict in enumerate(conflicts, 1):
        response += f"### {i}. {conflict.get('subject', 'Unknown Subject').title()}\n\n"
        response += f"**Type:** {conflict.get('type', 'potential_inconsistency').replace('_', ' ').title()}\n\n"
        
        if conflict.get("description"):
            response += f"{conflict['description']}\n\n"
        
        response += "**Conflicting Claims:**\n\n"
        for claim in conflict.get("claims", []):
            source = claim.get("source", "Unknown source")
            claim_text = claim.get("claim", "No claim text")
            quote = claim.get("quote", "")
            
            response += f"**{source}:** {claim_text}\n\n"
            if quote:
                response += f"> \"{quote[:200]}{'...' if len(quote) > 200 else ''}\"\n\n"
        
        response += "---\n\n"
    
    return (response, True)


def _answer_entities_query(question: str, data: dict) -> Tuple[str, bool]:
    """Generate response for entity queries - both specific lookups and listing."""
    
    entities = data.get("entities", [])
    claims = data.get("claims", [])
    question_lower = question.lower()
    
    # First, check if this is a specific entity lookup
    potential_names = _extract_potential_names(question)
    if potential_names:
        matching_entities = _find_matching_entities(potential_names, entities)
        
        if matching_entities:
            return _answer_specific_entity_query(question, matching_entities, claims, data)
        else:
            # Names were mentioned but NOT found in documents - fail gracefully
            names_str = ", ".join(f'"{name}"' for name in potential_names)
            return (
                f"## No Information Found\n\n"
                f"I searched the documents but could not find any information about {names_str}.\n\n"
                f"The following people ARE mentioned in the documents:\n\n" +
                "\n".join(f"- **{e.get('name')}**" for e in entities if e.get('type', '').lower() == 'person')[:15] +
                "\n\n*If you're looking for someone specific, please check the spelling or try a different name.*",
                True
            )
    
    # Otherwise, handle as a listing query
    # Filter by type if specified
    if any(kw in question_lower for kw in ["people", "person", "everyone", "names", "name", "individuals", "suspects", "witnesses", "victims"]):
        filtered = [e for e in entities if e.get("type", "").lower() == "person"]
        entity_type = "People"
    elif "organization" in question_lower or "compan" in question_lower:
        filtered = [e for e in entities if e.get("type", "").lower() == "organization"]
        entity_type = "Organizations"
    elif "location" in question_lower or "place" in question_lower:
        filtered = [e for e in entities if e.get("type", "").lower() == "location"]
        entity_type = "Locations"
    else:
        filtered = entities
        entity_type = "Entities"
    
    if not filtered:
        return (f"No {entity_type.lower()} were identified in the documents.", True)
    
    # Deduplicate by name (case-insensitive)
    seen_names = set()
    unique_entities = []
    for e in filtered:
        name_lower = e.get("name", "").lower()
        if name_lower and name_lower not in seen_names:
            seen_names.add(name_lower)
            unique_entities.append(e)
    
    response = f"## {entity_type} Mentioned in Documents\n\n"
    response += f"Found **{len(unique_entities)}** unique {entity_type.lower()}:\n\n"
    
    for entity in unique_entities:
        name = entity.get("name", "Unknown")
        desc = entity.get("description", "")
        source = entity.get("source", "Unknown source")
        entity_type_str = entity.get("type", "")
        
        response += f"### {name}"
        if entity_type_str and entity_type_str.lower() != entity_type.lower().rstrip('s'):
            response += f" ({entity_type_str})"
        response += "\n\n"
        
        if desc:
            response += f"{desc}\n\n"
        
        # Format source
        if isinstance(source, list):
            response += f"*Sources: {', '.join(source)}*\n\n"
        else:
            response += f"*Source: {source}*\n\n"
    
    return (response, True)


def _answer_specific_entity_query(
    question: str, 
    matching_entities: List[dict], 
    claims: List[dict],
    data: dict
) -> Tuple[str, bool]:
    """
    Generate response for a specific entity lookup.
    Aggregates entity info, related claims, and events.
    """
    
    response = ""
    
    for entity in matching_entities:
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "")
        desc = entity.get("description", "")
        source = entity.get("source", "Unknown source")
        mentions = entity.get("mentions", [])
        
        response += f"## {name}"
        if entity_type:
            response += f" ({entity_type})"
        response += "\n\n"
        
        if desc:
            response += f"{desc}\n\n"
        
        # Format source(s)
        if isinstance(source, list):
            response += f"**Sources:** {', '.join(source)}\n\n"
        else:
            response += f"**Source:** {source}\n\n"
        
        # Add mentions/quotes if available
        if mentions:
            response += "### Direct Mentions\n\n"
            for mention in mentions[:5]:  # Limit to 5
                if mention:
                    response += f"> \"{mention[:300]}{'...' if len(mention) > 300 else ''}\"\n\n"
        
        # Find related claims about this entity
        entity_name_lower = _normalize_for_matching(name)
        entity_words = set(entity_name_lower.split())
        
        related_claims = []
        for claim in claims:
            subject = claim.get("subject", "").lower()
            claim_text = claim.get("claim", "").lower()
            
            # Check if entity is mentioned in subject or claim
            if entity_name_lower in subject or entity_name_lower in claim_text:
                related_claims.append(claim)
            elif entity_words and any(word in subject or word in claim_text for word in entity_words):
                related_claims.append(claim)
        
        if related_claims:
            response += "### Related Claims\n\n"
            for claim in related_claims[:10]:  # Limit to 10
                claim_text = claim.get("claim", "")
                claim_source = claim.get("source", "Unknown")
                quote = claim.get("quote", "")
                
                response += f"**{claim_text}**\n\n"
                if quote:
                    response += f"> \"{quote[:200]}{'...' if len(quote) > 200 else ''}\"\n\n"
                response += f"*Source: {claim_source}*\n\n---\n\n"
        
        # Find related events
        events = data.get("events", [])
        related_events = []
        for event in events:
            people = [p.lower() for p in event.get("people_involved", [])]
            desc_lower = event.get("description", "").lower()
            
            if entity_name_lower in desc_lower or any(word in desc_lower for word in entity_words):
                related_events.append(event)
            elif any(_normalize_for_matching(name) in p for p in people for name_part in entity_words):
                related_events.append(event)
        
        if related_events:
            response += "### Related Events\n\n"
            for event in related_events[:5]:  # Limit to 5
                date = event.get("date", "Unknown date")
                event_desc = event.get("description", "")
                event_source = event.get("source", "Unknown")
                
                response += f"**{date}**\n\n{event_desc}\n\n*Source: {event_source}*\n\n---\n\n"
    
    if not response:
        return ("No information found for the specified entity.", False)
    
    return (response, True)


def _answer_events_query(data: dict) -> Tuple[str, bool]:
    """Generate response for timeline/events queries."""
    
    events = data.get("events", [])
    
    if not events:
        return ("No dated events were identified in the documents.", True)
    
    # Sort by date if possible
    def sort_key(e):
        date = e.get("date", "")
        if date and date.lower() != "unknown":
            return (0, date)
        return (1, "")
    
    sorted_events = sorted(events, key=sort_key)
    
    response = "## Timeline of Events\n\n"
    response += f"Found **{len(events)}** events:\n\n"
    
    for event in sorted_events:
        date = event.get("date", "Unknown date")
        desc = event.get("description", "No description")
        people = event.get("people_involved", [])
        location = event.get("location", "")
        source = event.get("source", "Unknown source")
        
        response += f"### {date}\n\n"
        response += f"{desc}\n\n"
        
        if people:
            response += f"- **People involved:** {', '.join(people)}\n"
        if location:
            response += f"- **Location:** {location}\n"
        
        response += f"\n*Source: {source}*\n\n---\n\n"
    
    return (response, True)


def _answer_summary_query(data: dict) -> Tuple[str, bool]:
    """Generate response for summary queries."""
    
    summary = get_extraction_summary(data)
    key_facts = data.get("key_facts", [])
    
    response = "## Document Summary\n\n"
    response += "### Overview\n\n"
    response += f"- **Documents Analyzed:** {summary['documents']}\n"
    response += f"- **Entities Identified:** {summary['entities']}\n"
    response += f"- **Claims Extracted:** {summary['claims']}\n"
    response += f"- **Events Found:** {summary['events']}\n"
    response += f"- **Potential Conflicts:** {summary['conflicts']}\n\n"
    
    if key_facts:
        response += "### Key Facts\n\n"
        for fact in key_facts[:20]:  # Limit to 20
            if isinstance(fact, dict):
                response += f"- {fact.get('fact', str(fact))} *({fact.get('source', 'unknown')})*\n"
            else:
                response += f"- {fact}\n"
        
        response += "\n"
        if len(key_facts) > 20:
            response += f"*...and {len(key_facts) - 20} more facts*\n\n"
    
    # Add entity breakdown
    entities = data.get("entities", [])
    if entities:
        response += "### Entity Breakdown\n\n"
        by_type = {}
        for e in entities:
            t = e.get("type", "Other")
            by_type[t] = by_type.get(t, 0) + 1
        
        for entity_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            response += f"- **{entity_type}:** {count}\n"
        
        response += "\n"
    
    return (response, True)


def _answer_general_exhaustive(question: str, data: dict) -> Tuple[str, bool]:
    """Fallback for general exhaustive queries."""
    
    summary = get_extraction_summary(data)
    
    response = "## Extracted Data Overview\n\n"
    response += f"I have preprocessed data from **{summary['documents']}** document(s):\n\n"
    response += f"- **{summary['entities']}** entities (people, organizations, locations)\n"
    response += f"- **{summary['claims']}** claims/statements\n"
    response += f"- **{summary['events']}** events\n"
    response += f"- **{summary['conflicts']}** potential conflicts detected\n\n"
    
    response += "For more specific information, try asking:\n"
    response += "- \"List all people mentioned\"\n"
    response += "- \"Show me the timeline of events\"\n"
    response += "- \"Find all inconsistencies\"\n"
    response += "- \"Give me a summary of all documents\"\n"
    
    return (response, True)


def should_use_extracted_data(question: str, extracted_data: dict = None) -> bool:
    """
    Quick check if a question should use extracted data (knowledge graph).
    Used to determine routing before full classification.
    
    Args:
        question: The user's question
        extracted_data: Optional pre-loaded extracted data (avoids re-loading)
    
    Returns:
        True if query should use knowledge graph, False for vector search
    """
    return classify_query(question, extracted_data) == "EXHAUSTIVE"

