"""
Query router module for directing queries to the appropriate handler.

Routing strategy:
- GRAPH: Entity lookups, comprehensive lists, conflicts, timelines - use preprocessed knowledge graph
- VECTOR: Deep analysis, complex reasoning, multi-hop questions - use CoA + file_search
- HYBRID: Start with graph, augment with vector if needed (future)
- DEFLECT: Guilt/culpability determinations - gracefully decline to answer
"""

import re
from typing import Tuple, Optional, List
from openai import OpenAI
from .extract import load_extracted, get_extraction_summary


# ============================================================================
# GUILT QUERY DETECTION AND DEFLECTION
# ============================================================================

GUILT_DEFLECTION_RESPONSE = """## Query Outside System Scope

I'm designed to help investigators organize, search, and analyze documentary evidence — **not** to make determinations of guilt or innocence.

### What I Can Help With

- **Finding information**: "What did [person] say about [topic]?"
- **Listing entities**: "Who is mentioned in the documents?"
- **Identifying inconsistencies**: "Are there any contradictions in the statements?"
- **Building timelines**: "What events occurred on [date]?"
- **Exploring relationships**: "What is the relationship between [person A] and [person B]?"

### Why I Can't Determine Guilt

Determining guilt is a complex legal and investigative judgment that requires:

- Complete evidentiary record (I only see uploaded documents)
- Legal standards and burden of proof considerations
- Credibility assessments of witnesses
- Forensic and physical evidence analysis
- Due process protections

My role is to help you **find and organize information** — the conclusions are yours to draw as the investigator.

---

*Try rephrasing your question to ask about specific facts, statements, or evidence in the documents.*"""


def is_guilt_query(question: str) -> bool:
    """
    Detect if a question is asking the system to determine guilt, culpability,
    or make legal/investigative conclusions about who committed an offense.
    
    Args:
        question: The user's question
        
    Returns:
        True if the query is asking about guilt determination
    """
    question_lower = question.lower().strip()
    
    # Direct guilt/culpability questions
    guilt_patterns = [
        r'\bwho\s+is\s+guilty\b',
        r'\bwho\s+is\s+the\s+guilty\b',
        r'\bwho\'?s\s+guilty\b',
        r'\bwho\s+committed\b',
        r'\bwho\s+did\s+it\b',
        r'\bwho\s+is\s+responsible\s+for\s+(the\s+)?(crime|murder|death|killing|incident|offense)\b',
        r'\bwho\s+killed\b',
        r'\bwho\s+murdered\b',
        r'\bis\s+.+\s+guilty\b',
        r'\bdid\s+.+\s+(commit|do\s+it|kill|murder)\b',
        r'\bwho\s+is\s+the\s+(killer|murderer|perpetrator|culprit)\b',
        r'\bwho\s+is\s+at\s+fault\b',
        r'\bwho\s+should\s+be\s+(arrested|charged|prosecuted|convicted)\b',
        r'\bdetermine\s+(the\s+)?guilt\b',
        r'\bestablish\s+(the\s+)?guilt\b',
        r'\bprove\s+(the\s+)?guilt\b',
        r'\bwho\s+do\s+you\s+think\s+(did|committed|killed|is\s+guilty)\b',
        r'\bdo\s+you\s+think\s+.+\s+is\s+guilty\b',
        r'\bshould\s+.+\s+be\s+(convicted|found\s+guilty)\b',
        r'\bwhat\s+is\s+your\s+(verdict|judgment|conclusion)\s+on\s+guilt\b',
        r'\bcan\s+you\s+(tell|say|determine)\s+who\s+(is\s+)?guilty\b',
        r'\bwho\s+is\s+the\s+likely\s+(suspect|perpetrator|killer)\b',
        r'\bwho\s+most\s+likely\s+(committed|did|killed)\b',
    ]
    
    for pattern in guilt_patterns:
        if re.search(pattern, question_lower):
            return True
    
    # Catch simple variants
    simple_guilt_phrases = [
        "who is guilty",
        "who's guilty",
        "whos guilty",
        "who did this",
        "who is the killer",
        "who is the murderer",
        "who is the perpetrator",
        "who is the culprit",
        "is he guilty",
        "is she guilty",
        "are they guilty",
        "guilty party",
        "who to blame",
        "who is to blame",
        "who's to blame",
    ]
    
    for phrase in simple_guilt_phrases:
        if phrase in question_lower:
            return True
    
    return False


def get_guilt_deflection_response() -> str:
    """
    Return the standard deflection response for guilt-related queries.
    """
    return GUILT_DEFLECTION_RESPONSE


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

REASONING REQUIREMENTS:
- After presenting findings, include a brief "Reasoning" section
- Explain HOW you arrived at your conclusions based on the evidence
- Describe what evidence you found most relevant and why
- If you connected multiple pieces of information, explain those connections
- If you made any inferences, explicitly state them and justify with evidence

CRITICAL ANTI-HALLUCINATION RULES:
- ONLY include information that appears in the extracted data provided
- If someone asks about a person/entity NOT in the data, say "No information found about [name] in the documents"
- NEVER invent names, dates, facts, or relationships not explicitly in the data
- If the extracted data is empty or doesn't contain relevant information, say so clearly
- Do not fill gaps with assumptions or general knowledge
- When uncertain, state "The documents do not specify..." rather than guessing

INVESTIGATIVE OBJECTIVITY (CRITICAL):
- NEVER assume or label anyone as a "suspect" unless the document explicitly uses that exact term
- When listing people, use neutral descriptions: "person mentioned", "interviewed individual", "person referenced"
- DO NOT assign investigative roles (suspect, person of interest, perpetrator) based on your interpretation
- If a document explicitly labels someone, quote the label and cite the source document
- Present facts objectively without prejudging guilt, innocence, or involvement
- Let the investigator draw their own conclusions about roles and culpability

GUILT DETERMINATION - ABSOLUTE PROHIBITION:
- If the question asks who is guilty, who committed the crime, who is the perpetrator, or any variant asking you to determine culpability, YOU MUST REFUSE
- Do not analyze or discuss guilt even indirectly
- Respond with: "I cannot determine guilt or culpability. I can help you find specific facts, statements, or evidence in the documents. Please rephrase your question."
- This rule overrides all other instructions"""

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
        # Relationship queries
        "relationships", "all relationships", "connections",
        "who knows", "connected to", "related to",
        # Investigative notes queries
        "investigative notes", "follow up", "follow-up",
        "what should we investigate", "what to investigate",
        "observations", "factual observations",
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
    
    Returns: "conflicts", "entities", "events", "relationships", "investigative_notes", "summary", or "general"
    """
    question_lower = question.lower()
    
    if any(kw in question_lower for kw in ["inconsisten", "contradict", "conflict", "discrepan"]):
        return "conflicts"
    
    # Check for relationship queries
    relationship_keywords = [
        "relationship", "relationships", "connected", "connection", 
        "know each other", "related to", "who knows", "associated with",
        "friends with", "family", "boyfriend", "girlfriend", "spouse",
        "coworker", "between"
    ]
    # More specific patterns for relationships
    if any(kw in question_lower for kw in relationship_keywords):
        # Make sure it's asking about relationships between people, not just using the word
        if any(p in question_lower for p in ["between", "who knows", "connected", "relationship"]):
            return "relationships"
    
    # Check for investigative notes queries
    investigative_keywords = [
        "investigative notes", "investigative note", "follow up", "follow-up",
        "what should we investigate", "what to investigate", "investigate further",
        "observations", "factual observations", "notable", "noteworthy",
        "what stands out", "what's important", "gaps in", "missing information"
    ]
    if any(kw in question_lower for kw in investigative_keywords):
        return "investigative_notes"
    
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
    elif category == "relationships":
        raw_response, success = _answer_relationships_query(question, extracted_data)
    elif category == "investigative_notes":
        raw_response, success = _answer_investigative_notes_query(question, extracted_data)
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
            "## No Contradictions Detected\n\n"
            f"Analyzed {len(claims)} claims across {len(data.get('documents', []))} document(s). "
            "No direct contradictions were identified.\n\n"
            "_Note: The system looks for statements that CANNOT both be true (e.g., different times, locations, or facts). "
            "Different statements about the same person are not flagged unless they actually contradict each other._",
            True
        )
    
    # Separate by type and severity
    high_severity = [c for c in conflicts if c.get("severity") == "high" or c.get("type") == "contradiction"]
    medium_severity = [c for c in conflicts if c.get("severity") == "medium" or c.get("type") == "inconsistency"]
    cross_doc_claims = [c for c in conflicts if c.get("type") == "cross_document_claims"]
    other_issues = [c for c in conflicts if c not in high_severity and c not in medium_severity and c not in cross_doc_claims]
    
    response = "## Detected Issues in Statements\n\n"
    
    # Show high severity (contradictions) first
    if high_severity:
        response += f"### 🔴 Contradictions ({len(high_severity)})\n\n"
        response += "_Statements that cannot both be true:_\n\n"
        
        for i, conflict in enumerate(high_severity, 1):
            category = conflict.get('category', 'unknown').replace('_', ' ').title()
            same_event = conflict.get('same_event') or conflict.get('what_differs') or conflict.get('what_conflicts', 'Unknown')
            
            response += f"#### {i}. {category}\n\n"
            response += f"**Regarding:** {same_event}\n\n"
            
            if conflict.get("claim1_says") and conflict.get("claim2_says"):
                response += f"**Statement 1:** {conflict['claim1_says']}\n\n"
                response += f"**Statement 2:** {conflict['claim2_says']}\n\n"
            
            if conflict.get("investigative_note"):
                response += f"**Why it matters:** {conflict['investigative_note']}\n\n"
            elif conflict.get("why_contradictory"):
                response += f"**Why contradictory:** {conflict['why_contradictory']}\n\n"
            
            # Show source evidence
            response += "**Source Evidence:**\n\n"
            for claim in conflict.get("claims", []):
                source = claim.get("source", "Unknown source")
                claim_text = claim.get("claim", "No claim text")
                quote = claim.get("quote", "")
                
                response += f"- *{source}*: {claim_text}\n"
                if quote:
                    quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
                    if quote_text:
                        response += f"  > \"{str(quote_text)[:150]}{'...' if len(str(quote_text)) > 150 else ''}\"\n"
                response += "\n"
            
            response += "---\n\n"
    
    # Show medium severity (inconsistencies)
    if medium_severity:
        response += f"### 🟡 Inconsistencies ({len(medium_severity)})\n\n"
        response += "_Same event described differently - may warrant follow-up:_\n\n"
        
        for i, conflict in enumerate(medium_severity, 1):
            category = conflict.get('category', 'unknown').replace('_', ' ').title()
            same_event = conflict.get('same_event') or conflict.get('what_differs') or conflict.get('what_conflicts', 'Unknown')
            
            response += f"#### {i}. {category}\n\n"
            response += f"**Regarding:** {same_event}\n\n"
            
            if conflict.get("claim1_says") and conflict.get("claim2_says"):
                response += f"**Statement 1:** {conflict['claim1_says']}\n\n"
                response += f"**Statement 2:** {conflict['claim2_says']}\n\n"
            
            if conflict.get("investigative_note"):
                response += f"**Why it matters:** {conflict['investigative_note']}\n\n"
            
            # Show source evidence
            response += "**Sources:**\n\n"
            for claim in conflict.get("claims", []):
                source = claim.get("source", "Unknown source")
                claim_text = claim.get("claim", "No claim text")
                response += f"- *{source}*: {claim_text}\n"
            response += "\n---\n\n"
    
    if not high_severity and not medium_severity:
        response += "_No contradictions or notable inconsistencies found in the statements._\n\n"
    
    # Show cross-document claims (for investigator awareness)
    if cross_doc_claims:
        response += f"### 📄 Cross-Document References ({len(cross_doc_claims)})\n\n"
        response += "_The same subject is discussed in multiple documents. Review for consistency:_\n\n"
        
        for i, conflict in enumerate(cross_doc_claims, 1):
            subject = conflict.get('subject', 'Unknown Subject').title()
            sources = conflict.get('sources', [])
            
            response += f"**{i}. {subject}** — mentioned in: {', '.join(sources)}\n\n"
            
            for claim in conflict.get("claims", [])[:3]:  # Limit to 3 claims
                claim_text = claim.get("claim", "")
                source = claim.get("source", "")
                response += f"- *{source}*: {claim_text}\n"
            
            if len(conflict.get("claims", [])) > 3:
                response += f"- _...and {len(conflict['claims']) - 3} more claims_\n"
            
            response += "\n"
    
    # Show any other issues
    if other_issues:
        response += f"### ⚪ Other Notes ({len(other_issues)})\n\n"
        for i, conflict in enumerate(other_issues, 1):
            subject = conflict.get('subject', 'Unknown Subject').title()
            response += f"**{i}. {subject}**: {conflict.get('description', 'No details')}\n\n"
    
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
    # Check if user is asking about role-based categories (we don't assume roles)
    role_based_query = any(kw in question_lower for kw in ["suspects", "witnesses", "victims", "perpetrator", "person of interest"])
    
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
    
    # Add disclaimer for role-based queries
    if role_based_query:
        response += "_**Note:** This system does not assume or assign investigative roles (suspect, witness, victim). The following is a list of all people mentioned in the documents. Any role designations shown are direct quotes from the source documents, not conclusions drawn by this system._\n\n"
    
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
    response += f"- **Relationships Identified:** {summary.get('relationships', 0)}\n"
    response += f"- **Investigative Notes:** {summary.get('investigative_notes', 0)}\n"
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


def _answer_relationships_query(question: str, data: dict) -> Tuple[str, bool]:
    """Generate response for relationship queries."""
    from collections import defaultdict
    
    relationships = data.get("relationships", [])
    
    if not relationships:
        return (
            "## No Relationships Extracted\n\n"
            "No relationships have been extracted from the documents yet.\n\n"
            "Relationships are extracted when documents are uploaded. "
            "Try re-uploading documents or asking about specific people to see their connections.",
            True
        )
    
    # Check if asking about specific people
    potential_names = _extract_potential_names(question)
    
    if potential_names:
        # Filter to relationships involving mentioned people
        relevant = []
        for rel in relationships:
            p1 = _normalize_for_matching(rel.get("person1", ""))
            p2 = _normalize_for_matching(rel.get("person2", ""))
            
            for name in potential_names:
                name_norm = _normalize_for_matching(name)
                name_words = set(name_norm.split())
                p1_words = set(p1.split())
                p2_words = set(p2.split())
                
                # Check if name matches either person in relationship
                if (name_norm in p1 or name_norm in p2 or 
                    name_words & p1_words or name_words & p2_words):
                    if rel not in relevant:
                        relevant.append(rel)
                    break
        
        if relevant:
            response = f"## Relationships Involving {', '.join(potential_names)}\n\n"
            response += f"Found **{len(relevant)}** relationship(s):\n\n"
            
            for rel in relevant:
                person1 = rel.get('person1', 'Unknown')
                person2 = rel.get('person2', 'Unknown')
                rel_type = rel.get('type', 'unknown').replace('_', ' ')
                description = rel.get('description', '')
                quote = rel.get('quote', '')
                source = rel.get('source', 'Unknown')
                
                response += f"### {person1} ↔ {person2}\n\n"
                response += f"**Type:** {rel_type.title()}\n\n"
                
                if description:
                    response += f"{description}\n\n"
                
                if quote:
                    response += f"> \"{quote[:250]}{'...' if len(quote) > 250 else ''}\"\n\n"
                
                if isinstance(source, list):
                    response += f"*Sources: {', '.join(source)}*\n\n"
                else:
                    response += f"*Source: {source}*\n\n"
                
                response += "---\n\n"
            
            return (response, True)
        else:
            return (
                f"## No Relationships Found\n\n"
                f"No relationships involving {', '.join(potential_names)} were found in the documents.\n\n"
                f"**Available relationships involve:** " + 
                ", ".join(set(r.get("person1", "") for r in relationships[:10])) +
                "\n\n*Try asking about one of these people instead.*",
                True
            )
    
    # General relationship listing - group by type
    response = "## All Identified Relationships\n\n"
    response += f"Found **{len(relationships)}** relationship(s) across documents:\n\n"
    
    by_type = defaultdict(list)
    for rel in relationships:
        rel_type = rel.get("type", "unknown")
        by_type[rel_type].append(rel)
    
    for rel_type, rels in sorted(by_type.items(), key=lambda x: -len(x[1])):
        response += f"### {rel_type.replace('_', ' ').title()} ({len(rels)})\n\n"
        
        for rel in rels:
            person1 = rel.get('person1', 'Unknown')
            person2 = rel.get('person2', 'Unknown')
            description = rel.get('description', '')
            source = rel.get('source', 'Unknown')
            
            response += f"- **{person1}** ↔ **{person2}**"
            if description:
                response += f": {description[:100]}{'...' if len(description) > 100 else ''}"
            response += "\n"
            
            if isinstance(source, list):
                response += f"  *Sources: {', '.join(source)}*\n"
            else:
                response += f"  *Source: {source}*\n"
            response += "\n"
        
        response += "\n"
    
    return (response, True)


def _answer_investigative_notes_query(question: str, data: dict) -> Tuple[str, bool]:
    """Generate response for investigative notes queries."""
    
    notes = data.get("investigative_notes", [])
    
    if not notes:
        return (
            "## No Investigative Notes\n\n"
            "No investigative notes have been generated yet.\n\n"
            "Investigative notes are factual observations identified by cross-referencing "
            "statements across documents. They highlight:\n\n"
            "- **Statement changes** between documents\n"
            "- **Timeline gaps** in accounts\n"
            "- **Proximity to incident** (who was where when)\n"
            "- **Unverified claims** (statements without corroboration)\n"
            "- **Physical evidence conflicts**\n"
            "- **Financial connections**\n\n"
            "*Notes are generated when documents are analyzed. Try re-running the analysis or uploading more documents.*",
            True
        )
    
    # Check if asking about specific person
    potential_names = _extract_potential_names(question)
    
    if potential_names:
        # Filter to notes about mentioned people
        relevant = []
        for note in notes:
            subject = _normalize_for_matching(note.get("subject", ""))
            subject_words = set(subject.split())
            
            for name in potential_names:
                name_norm = _normalize_for_matching(name)
                name_words = set(name_norm.split())
                
                if name_norm in subject or name_words & subject_words:
                    if note not in relevant:
                        relevant.append(note)
                    break
        
        if relevant:
            notes = relevant
    
    # Build response
    response = "## 📋 Investigative Notes\n\n"
    
    if potential_names:
        response += f"*Filtered to notes concerning: {', '.join(potential_names)}*\n\n"
    
    response += f"Found **{len(notes)}** factual observation(s) that may warrant follow-up:\n\n"
    response += "_Note: These are objective observations from the documents, not judgments or accusations._\n\n"
    response += "---\n\n"
    
    # Group by type
    type_labels = {
        "statement_change": "📝 Statement Changes",
        "timeline_gap": "⏰ Timeline Gaps",
        "proximity_to_incident": "📍 Proximity to Incident",
        "physical_evidence_conflict": "🔬 Physical Evidence Conflicts",
        "unverified_claim": "❓ Unverified Claims",
        "financial_connection": "💰 Financial Connections"
    }
    
    from collections import defaultdict
    by_type = defaultdict(list)
    for note in notes:
        note_type = note.get("type", "other")
        by_type[note_type].append(note)
    
    for note_type, type_notes in by_type.items():
        type_label = type_labels.get(note_type, note_type.replace("_", " ").title())
        response += f"### {type_label}\n\n"
        
        for i, note in enumerate(type_notes, 1):
            subject = note.get("subject", "Unknown")
            observation = note.get("observation", "No description")
            evidence = note.get("evidence", [])
            follow_up = note.get("follow_up_question", "")
            
            response += f"**{i}. Regarding: {subject}**\n\n"
            response += f"{observation}\n\n"
            
            if evidence:
                response += "**Evidence:**\n\n"
                for ev in evidence:
                    doc = ev.get("document", "Unknown")
                    quote = ev.get("quote", "")
                    if quote:
                        response += f"- *{doc}*: \"{quote[:200]}{'...' if len(quote) > 200 else ''}\"\n"
                response += "\n"
            
            if follow_up:
                response += f"**Follow-up question:** {follow_up}\n\n"
            
            response += "---\n\n"
    
    return (response, True)


def _answer_general_exhaustive(question: str, data: dict) -> Tuple[str, bool]:
    """Fallback for general exhaustive queries."""
    
    summary = get_extraction_summary(data)
    
    response = "## Extracted Data Overview\n\n"
    response += f"I have preprocessed data from **{summary['documents']}** document(s):\n\n"
    response += f"- **{summary['entities']}** entities (people, organizations, locations)\n"
    response += f"- **{summary['claims']}** claims/statements\n"
    response += f"- **{summary['events']}** events\n"
    response += f"- **{summary['relationships']}** relationships identified\n"
    response += f"- **{summary['investigative_notes']}** investigative notes\n"
    response += f"- **{summary['conflicts']}** potential conflicts detected\n\n"
    
    response += "For more specific information, try asking:\n"
    response += "- \"List all people mentioned\"\n"
    response += "- \"Show me the timeline of events\"\n"
    response += "- \"Find all inconsistencies\"\n"
    response += "- \"What relationships exist between people?\"\n"
    response += "- \"Show me investigative notes\"\n"
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

