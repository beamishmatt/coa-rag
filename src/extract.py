"""
Document extraction module for preprocessing documents at upload time.
Extracts structured data (entities, claims, events) for exhaustive queries.
"""

import json
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

EXTRACTED_PATH = Path("data/extracted.json")


def load_extracted() -> dict:
    """Load previously extracted data from JSON file."""
    if EXTRACTED_PATH.exists():
        try:
            return json.loads(EXTRACTED_PATH.read_text())
        except json.JSONDecodeError:
            return _empty_extraction()
    return _empty_extraction()


def _empty_extraction() -> dict:
    """Return empty extraction structure."""
    return {
        "entities": [],
        "claims": [],
        "events": [],
        "conflicts": [],
        "documents": []
    }


def save_extracted(data: dict):
    """Save extracted data to JSON file."""
    EXTRACTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXTRACTED_PATH.write_text(json.dumps(data, indent=2))


def extract_from_document(client: OpenAI, model: str, doc_text: str, doc_name: str) -> dict:
    """
    Extract structured information from a document using LLM.
    Run once per document at upload time.
    """
    
    prompt = """Analyze this document and extract ALL structured information.

Return ONLY valid JSON with this exact structure:
{
    "entities": [
        {"name": "full name or title", "type": "Person|Organization|Location|Date|Money|Other", "description": "brief context about this entity", "mentions": ["quote where mentioned"]}
    ],
    "claims": [
        {"subject": "who or what the claim is about", "claim": "what is being stated/claimed", "quote": "exact quote from document", "context": "surrounding context"}
    ],
    "events": [
        {"date": "date if mentioned or 'unknown'", "description": "what happened", "people_involved": ["names"], "location": "where if mentioned"}
    ],
    "key_facts": [
        "important factual statements from the document"
    ]
}

Rules:
- Extract ALL people, organizations, locations, dates, and monetary amounts mentioned
- Include exact quotes where possible
- For claims, focus on assertions, statements, and testimony
- For events, capture anything with a temporal or sequential nature
- Be thorough - this extraction will be used to answer comprehensive queries later

DOCUMENT:
"""
    
    try:
        resp = client.responses.create(
            model=model,
            input=f"{prompt}\n\n{doc_text[:50000]}"  # Limit to ~50k chars to stay within context
        )
        
        # Try to parse JSON from response
        response_text = resp.output_text.strip()
        
        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (code fence)
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
        
        extracted = json.loads(response_text)
        
        # Tag everything with source document
        for item in extracted.get("entities", []):
            item["source"] = doc_name
        for item in extracted.get("claims", []):
            item["source"] = doc_name
        for item in extracted.get("events", []):
            item["source"] = doc_name
        for i, fact in enumerate(extracted.get("key_facts", [])):
            if isinstance(fact, str):
                extracted["key_facts"][i] = {"fact": fact, "source": doc_name}
        
        return extracted
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error during extraction: {e}")
        return {
            "entities": [],
            "claims": [],
            "events": [],
            "key_facts": [],
            "extraction_error": str(e),
            "raw_response": resp.output_text[:1000] if 'resp' in dir() else "No response"
        }
    except Exception as e:
        print(f"Extraction error: {e}")
        return {
            "entities": [],
            "claims": [],
            "events": [],
            "key_facts": [],
            "extraction_error": str(e)
        }


def merge_extraction(all_data: dict, new_extraction: dict, doc_name: str) -> dict:
    """Merge new extraction with existing data, deduplicating entities."""
    
    # Add document to list if not already present
    if doc_name not in all_data.get("documents", []):
        all_data.setdefault("documents", []).append(doc_name)
    
    # Merge entities with deduplication
    all_data.setdefault("entities", [])
    existing_entities = {_normalize_name(e.get("name", "")): i for i, e in enumerate(all_data["entities"])}
    
    for new_entity in new_extraction.get("entities", []):
        normalized_name = _normalize_name(new_entity.get("name", ""))
        if normalized_name in existing_entities:
            # Merge with existing entity
            idx = existing_entities[normalized_name]
            existing = all_data["entities"][idx]
            # Merge mentions
            existing_mentions = set(existing.get("mentions", []))
            new_mentions = new_entity.get("mentions", [])
            existing["mentions"] = list(existing_mentions | set(new_mentions))
            # Add source if different
            existing_source = existing.get("source", "")
            new_source = new_entity.get("source", "")
            if new_source and new_source != existing_source:
                if isinstance(existing_source, list):
                    if new_source not in existing_source:
                        existing_source.append(new_source)
                else:
                    existing["source"] = [existing_source, new_source] if existing_source else new_source
            # Keep longer description
            if len(new_entity.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = new_entity["description"]
        else:
            # Add new entity
            all_data["entities"].append(new_entity)
            existing_entities[normalized_name] = len(all_data["entities"]) - 1
    
    # Merge claims
    all_data.setdefault("claims", []).extend(new_extraction.get("claims", []))
    
    # Merge events
    all_data.setdefault("events", []).extend(new_extraction.get("events", []))
    
    # Merge key facts
    all_data.setdefault("key_facts", []).extend(new_extraction.get("key_facts", []))
    
    return all_data


def _normalize_name(name: str) -> str:
    """Normalize entity name for deduplication."""
    if not name:
        return ""
    # Lowercase, strip whitespace
    normalized = name.lower().strip()
    return normalized


def _names_match(name1: str, name2: str) -> bool:
    """Check if two names refer to the same entity."""
    n1 = _normalize_name(name1)
    n2 = _normalize_name(name2)
    
    if not n1 or not n2:
        return False
    
    # Exact match
    if n1 == n2:
        return True
    
    # One contains the other (e.g., "Amanda" in "Amanda Lynn Plasse")
    if n1 in n2 or n2 in n1:
        # Only match if it's a word boundary (not partial word)
        words1 = set(n1.split())
        words2 = set(n2.split())
        # If all words of one are in the other, it's a match
        if words1.issubset(words2) or words2.issubset(words1):
            return True
    
    return False


def deduplicate_entities(entities: list) -> list:
    """Deduplicate a list of entities, merging similar ones."""
    if not entities:
        return []
    
    merged = []
    
    for entity in entities:
        name = entity.get("name", "")
        entity_type = entity.get("type", "")
        
        # Find matching existing entity
        match_idx = None
        for i, existing in enumerate(merged):
            if existing.get("type", "") == entity_type and _names_match(name, existing.get("name", "")):
                match_idx = i
                break
        
        if match_idx is not None:
            # Merge with existing
            existing = merged[match_idx]
            # Merge mentions
            existing_mentions = set(existing.get("mentions", []))
            new_mentions = entity.get("mentions", [])
            existing["mentions"] = list(existing_mentions | set(new_mentions))
            # Merge sources
            existing_source = existing.get("source", "")
            new_source = entity.get("source", "")
            if new_source:
                if isinstance(existing_source, list):
                    if new_source not in existing_source:
                        existing_source.append(new_source)
                    existing["source"] = existing_source
                elif existing_source:
                    if new_source != existing_source:
                        existing["source"] = [existing_source, new_source]
                else:
                    existing["source"] = new_source
            # Keep the longer/more complete name
            if len(name) > len(existing.get("name", "")):
                existing["name"] = name
            # Keep longer description
            if len(entity.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = entity["description"]
        else:
            # Add as new
            merged.append(entity.copy())
    
    return merged


def deduplicate_extracted_data(all_data: dict) -> dict:
    """Deduplicate all entities in the extracted data."""
    all_data["entities"] = deduplicate_entities(all_data.get("entities", []))
    return all_data


def detect_conflicts(all_data: dict, client: OpenAI = None, model: str = None) -> list:
    """
    Detect potential conflicts/inconsistencies across claims from different documents.
    Uses simple heuristics first, then optionally LLM for deeper analysis.
    """
    
    conflicts = []
    claims = all_data.get("claims", [])
    
    if len(claims) < 2:
        return conflicts
    
    # Group claims by subject (normalized)
    by_subject = defaultdict(list)
    for claim in claims:
        subject = claim.get("subject", "").lower().strip()
        if subject:
            by_subject[subject].append(claim)
    
    # Find subjects with multiple claims from different sources
    for subject, subject_claims in by_subject.items():
        sources = set(c.get("source", "") for c in subject_claims)
        
        if len(sources) > 1 or len(subject_claims) > 1:
            # Multiple claims about the same subject - potential conflict
            unique_claim_texts = set(c.get("claim", "").lower() for c in subject_claims)
            
            if len(unique_claim_texts) > 1:
                conflicts.append({
                    "subject": subject,
                    "type": "potential_inconsistency",
                    "claims": subject_claims,
                    "sources": list(sources),
                    "description": f"Multiple different claims about '{subject}' found across documents"
                })
    
    # If we have an LLM client, do deeper conflict analysis
    if client and model and claims:
        llm_conflicts = _detect_conflicts_with_llm(client, model, claims)
        conflicts.extend(llm_conflicts)
    
    return conflicts


def _detect_conflicts_with_llm(client: OpenAI, model: str, claims: list) -> list:
    """Use LLM to detect semantic conflicts between claims."""
    
    if len(claims) < 2:
        return []
    
    # Limit claims to avoid token limits
    claims_sample = claims[:50]
    
    prompt = """Analyze these claims extracted from investigation documents.
Identify any CONTRADICTIONS or INCONSISTENCIES between claims.

Claims:
"""
    for i, claim in enumerate(claims_sample):
        prompt += f"\n{i+1}. [{claim.get('source', 'unknown')}] {claim.get('subject', 'unknown')}: {claim.get('claim', '')}"
        if claim.get('quote'):
            prompt += f' (Quote: "{claim.get("quote")[:100]}...")'
    
    prompt += """

Return ONLY valid JSON array of conflicts found:
[
    {
        "claim_indices": [index1, index2],
        "type": "contradiction|inconsistency|discrepancy",
        "description": "explanation of the conflict"
    }
]

If no conflicts found, return empty array: []
"""
    
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
        
        llm_conflicts = json.loads(response_text)
        
        # Enrich with actual claim data
        enriched = []
        for conflict in llm_conflicts:
            indices = conflict.get("claim_indices", [])
            conflict["claims"] = [claims_sample[i] for i in indices if i < len(claims_sample)]
            conflict["sources"] = list(set(c.get("source", "") for c in conflict["claims"]))
            enriched.append(conflict)
        
        return enriched
        
    except Exception as e:
        print(f"LLM conflict detection error: {e}")
        return []


def remove_document_extraction(doc_name: str) -> dict:
    """Remove all extracted data for a specific document."""
    all_data = load_extracted()
    
    # Remove document from list
    if doc_name in all_data.get("documents", []):
        all_data["documents"].remove(doc_name)
    
    # Filter out entities from this document
    all_data["entities"] = [e for e in all_data.get("entities", []) if e.get("source") != doc_name]
    
    # Filter out claims from this document
    all_data["claims"] = [c for c in all_data.get("claims", []) if c.get("source") != doc_name]
    
    # Filter out events from this document
    all_data["events"] = [e for e in all_data.get("events", []) if e.get("source") != doc_name]
    
    # Filter out key facts from this document
    all_data["key_facts"] = [f for f in all_data.get("key_facts", []) 
                            if isinstance(f, dict) and f.get("source") != doc_name]
    
    # Clear conflicts (will be recalculated)
    all_data["conflicts"] = []
    
    save_extracted(all_data)
    return all_data


def get_extraction_summary(all_data: dict = None) -> dict:
    """Get a summary of extracted data."""
    if all_data is None:
        all_data = load_extracted()
    
    return {
        "documents": len(all_data.get("documents", [])),
        "entities": len(all_data.get("entities", [])),
        "claims": len(all_data.get("claims", [])),
        "events": len(all_data.get("events", [])),
        "conflicts": len(all_data.get("conflicts", [])),
        "key_facts": len(all_data.get("key_facts", []))
    }

