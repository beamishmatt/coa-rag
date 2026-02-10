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
        "relationships": [],
        "investigative_notes": [],
        "conflicts": [],
        "documents": [],
        "key_facts": [],
        "benefit_indicators": []
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
    "document_date": "date when this document/interview was created (if stated)",
    "entities": [
        {"name": "full name or title", "type": "Person|Organization|Location|Date|Money|Other", "description": "brief context about this entity", "mentions": ["quote where mentioned"]}
    ],
    "claims": [
        {"subject": "who or what the claim is about", "claim": "what is being stated/claimed", "quote": "exact quote from document", "context": "surrounding context"}
    ],
    "events": [
        {"date": "when the event ACTUALLY HAPPENED or 'unknown'", "description": "what happened", "people_involved": ["names"], "location": "where if mentioned", "date_source": "how the date was determined"}
    ],
    "relationships": [
        {"person1": "name of first person", "person2": "name of second person", "type": "relationship type", "description": "context about the relationship", "quote": "supporting quote from document"}
    ],
    "key_facts": [
        "important factual statements from the document"
    ],
    "benefit_indicators": [
        {"person": "name of person who might benefit", "victim": "name of victim/deceased if applicable", "type": "financial|practical|emotional", "subtype": "insurance_beneficiary|inheritance|debt_relief|control_gain|obstacle_removed|secret_protected|jealousy|revenge|relationship_conflict", "description": "what benefit they might gain", "quote": "supporting evidence from document"}
    ]
}

CRITICAL DATE HANDLING - READ CAREFULLY:
There are TWO types of dates in documents:
1. DOCUMENT DATE: When the interview/statement was recorded (e.g., "Statement taken on November 1, 2013")
2. EVENT DATE: When the described events actually happened (often DIFFERENT from document date!)

For INTERVIEW TRANSCRIPTS and WITNESS STATEMENTS:
- The document_date is when the interview occurred
- Events described BY the interviewee happened at a DIFFERENT time (usually earlier)
- For each event, use the date WHEN THE EVENT ACTUALLY HAPPENED, NOT the interview date
- If the interviewee says "I delivered weed to her that day" referring to a past event, determine WHEN that day was
- Look for context clues: "the day she was killed", "that morning", "back in August", etc.
- If the exact event date is unclear, use "unknown" or contextual dating like "before [death date]" or "around August 2011"
- NEVER use the interview/document date as the event date unless the event literally occurred during the interview

VALIDATION: If an event involves interaction with a person who is deceased:
- The event MUST have occurred BEFORE that person's death
- If someone was killed on 08/26/2011, any interaction with them must be dated on or before 08/26/2011

RELATIONSHIP TYPES to look for:
- Family: parent, child, sibling, spouse, ex-spouse, relative
- Romantic: boyfriend, girlfriend, ex-boyfriend, ex-girlfriend
- Professional: coworker, employer, employee, business_partner
- Social: friend, acquaintance, neighbor, roommate
- Case-related: interviewed, mentioned_in_case, referenced, present_at_scene
- Transactional: buyer, seller, dealer, customer
- General: knows, associated_with

Rules:
- Extract ALL people, organizations, locations, dates, and monetary amounts mentioned
- Include exact quotes where possible
- For claims, focus on assertions, statements, and testimony
- For events, capture anything with a temporal or sequential nature
- For relationships, extract ANY connection between people mentioned - even indirect ones
- Be thorough - this extraction will be used to answer comprehensive queries later

CRITICAL - CLAIMS FROM INTERVIEWS ARE UNVERIFIED:
- Just because someone SAID something in an interview does NOT mean it happened
- When extracting claims from interviews, these are what the person CLAIMED, not verified facts
- People lie, misremember, have biases, and protect themselves or others
- The "claim" field should capture what was stated, with the understanding it may not be true
- Extract the claim faithfully, but understand it will be presented with appropriate hedging later

BENEFIT INDICATORS - Extract any evidence of who might benefit from the incident:
- FINANCIAL benefits: Insurance beneficiaries, inheritance, debts owed TO the victim (cleared by death), shared assets, business ownership changes
- PRACTICAL benefits: Obstacles removed (someone blocking their goals), control/power gained, secrets protected (victim knew something damaging)
- EMOTIONAL benefits: Jealousy (romantic rivals), revenge (prior conflict/grievance), relationship conflict resolution (custody, divorce disputes)

For each benefit indicator:
- "type" = financial, practical, or emotional
- "subtype" = specific category (insurance_beneficiary, inheritance, debt_relief, control_gain, obstacle_removed, secret_protected, jealousy, revenge, relationship_conflict)
- MUST include supporting quote from the document
- Be objective: extract documented facts, not speculative connections
- Only extract if there is actual textual evidence in the document

KEY_FACTS TEMPORAL CLARITY - CRITICAL:
When writing key_facts involving timing or sequence, use UNAMBIGUOUS phrasing:
- Write facts with the SUBJECT FIRST and explicit temporal markers
- WRONG: "X was killed after Y saw her" (ambiguous - can be misread as "Y saw X after X was killed")
- RIGHT: "Y's last contact with X occurred before X's death" (unambiguous)
- For deceased persons: ALWAYS phrase as "[Person]'s interaction with [victim] occurred before [victim]'s death"
- Never write temporal facts that could be misread if subject/object are mentally swapped
- Use "before [person]'s death" rather than "after [person] saw them" to avoid inversion errors

ENTITY LABELING - CRITICAL FOR PERSONS:
- NEVER describe someone as "Suspect" or "Suspect in the case" as your own classification
- If a document explicitly lists someone under a "Suspect(s):" field or similar label:
  → Description MUST be: "Listed under 'Suspect(s)' in police report" - quoting the document's exact label
  → This makes clear the label comes from the document, not from us
- For all other persons, use NEUTRAL descriptions based on their role:
  → "Person of interest; mentioned in connection with the case"
  → "Interviewed individual"
  → "Witness"
  → "Person mentioned in case documents"
  → "Reporting officer" (for law enforcement)
- The description should explain WHO the person is and their ROLE, not assign guilt or suspicion
- DO NOT use: "Suspect in the case", "Prime suspect", "Main suspect", or any variant
- DO use quoted labels when the document provides them: "Listed under '[exact field name]' in [document type]"

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
            # Also tag with document_date if available for context
            if extracted.get("document_date"):
                item["document_date"] = extracted["document_date"]
        for item in extracted.get("relationships", []):
            item["source"] = doc_name
        for i, fact in enumerate(extracted.get("key_facts", [])):
            if isinstance(fact, str):
                extracted["key_facts"][i] = {"fact": fact, "source": doc_name}
        for item in extracted.get("benefit_indicators", []):
            item["source"] = doc_name
        
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


def _parse_date(date_str: str) -> tuple:
    """
    Parse a date string into comparable components.
    Returns (year, month, day) or None if unparseable.
    """
    if not date_str or date_str.lower() in ['unknown', 'unclear', 'unspecified']:
        return None
    
    import re
    
    # Common date formats
    patterns = [
        # MM/DD/YYYY or M/D/YYYY
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2)))),
        # YYYY-MM-DD
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        # Month DD, YYYY (e.g., "August 26, 2011")
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', 
         lambda m: (int(m.group(3)), _month_to_num(m.group(1)), int(m.group(2)))),
        # DD Month YYYY (e.g., "26 August 2011")
        (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
         lambda m: (int(m.group(3)), _month_to_num(m.group(2)), int(m.group(1)))),
        # Just year
        (r'^(\d{4})$', lambda m: (int(m.group(1)), 1, 1)),
    ]
    
    for pattern, extractor in patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                return extractor(match)
            except (ValueError, IndexError):
                continue
    
    return None


def _month_to_num(month_name: str) -> int:
    """Convert month name to number."""
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    return months.get(month_name.lower(), 1)


def _date_after(date1: tuple, date2: tuple) -> bool:
    """Check if date1 is after date2. Both are (year, month, day) tuples."""
    if not date1 or not date2:
        return False
    return date1 > date2


def validate_event_dates(all_data: dict) -> dict:
    """
    Validate event dates against known death dates.
    Events involving deceased persons must predate their death.
    
    Returns the data with invalid dates flagged/corrected.
    """
    # Find death events to establish death dates
    death_dates = {}
    death_keywords = ['killed', 'murdered', 'death', 'died', 'fatal', 'homicide', 'deceased', 'body found', 'body discovered']
    
    events = all_data.get("events", [])
    entities = all_data.get("entities", [])
    
    # First pass: identify death dates from events
    for event in events:
        description = event.get("description", "").lower()
        if any(keyword in description for keyword in death_keywords):
            people = event.get("people_involved", [])
            date_str = event.get("date", "")
            parsed_date = _parse_date(date_str)
            
            if parsed_date:
                for person in people:
                    person_lower = person.lower()
                    # Check if this person is the victim (not just mentioned)
                    if any(v in description for v in ['victim', person_lower]):
                        death_dates[person_lower] = parsed_date
    
    # Also check entities for death dates
    for entity in entities:
        desc = entity.get("description", "").lower()
        if any(keyword in desc for keyword in ['deceased', 'victim', 'killed', 'murdered']):
            name = entity.get("name", "").lower()
            # Try to find an associated death date from mentions
            for mention in entity.get("mentions", []):
                parsed = _parse_date(mention)
                if parsed:
                    death_dates[name] = parsed
                    break
    
    # Second pass: validate all events
    validation_issues = []
    for i, event in enumerate(events):
        event_date = _parse_date(event.get("date", ""))
        if not event_date:
            continue
            
        people = event.get("people_involved", [])
        description = event.get("description", "").lower()
        
        # Skip death events themselves
        if any(keyword in description for keyword in death_keywords):
            continue
        
        for person in people:
            person_lower = person.lower()
            # Check against all known deceased persons (fuzzy match)
            for deceased_name, death_date in death_dates.items():
                # Check if names match (full name or partial)
                name_parts = person_lower.split()
                deceased_parts = deceased_name.split()
                
                # Match if any significant name part overlaps
                if (person_lower in deceased_name or 
                    deceased_name in person_lower or
                    any(part in deceased_parts for part in name_parts if len(part) > 2)):
                    
                    if _date_after(event_date, death_date):
                        # This is an impossible date!
                        issue = {
                            "event_index": i,
                            "event": event,
                            "person": person,
                            "event_date": event.get("date"),
                            "death_date": f"{death_date[1]}/{death_date[2]}/{death_date[0]}",
                            "issue": f"Event date ({event.get('date')}) is after {person}'s death"
                        }
                        validation_issues.append(issue)
                        
                        # Fix the event: mark date as needing review
                        event["date"] = f"unknown (before {death_date[1]}/{death_date[2]}/{death_date[0]})"
                        event["date_validation_issue"] = f"Original date '{event.get('date', 'unknown')}' was after {person}'s death - corrected"
                        event["date_source"] = "corrected - original date was interview date, not event date"
    
    if validation_issues:
        all_data.setdefault("validation_issues", []).extend(validation_issues)
        print(f"Found {len(validation_issues)} date validation issues")
        for issue in validation_issues:
            print(f"  - {issue['issue']}: {issue['event'].get('description', '')[:50]}...")
    
    return all_data


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
    
    # Merge relationships with deduplication
    all_data.setdefault("relationships", [])
    all_data["relationships"] = _merge_relationships(
        all_data["relationships"], 
        new_extraction.get("relationships", [])
    )
    
    # Merge key facts
    all_data.setdefault("key_facts", []).extend(new_extraction.get("key_facts", []))
    
    # Merge benefit indicators
    all_data.setdefault("benefit_indicators", []).extend(new_extraction.get("benefit_indicators", []))
    
    # Validate event dates against known death dates
    all_data = validate_event_dates(all_data)
    
    return all_data


def _merge_relationships(existing: list, new_relationships: list) -> list:
    """Merge relationships, avoiding duplicates."""
    
    def relationship_key(r):
        """Create a normalized key for relationship comparison."""
        names = sorted([
            _normalize_name(r.get("person1", "")), 
            _normalize_name(r.get("person2", ""))
        ])
        return (names[0], names[1], r.get("type", "").lower())
    
    existing_keys = {relationship_key(r) for r in existing}
    
    for rel in new_relationships:
        key = relationship_key(rel)
        
        if key not in existing_keys:
            existing.append(rel)
            existing_keys.add(key)
        else:
            # Find existing and merge sources if different
            for existing_rel in existing:
                if relationship_key(existing_rel) == key:
                    # Merge sources
                    existing_source = existing_rel.get("source", "")
                    new_source = rel.get("source", "")
                    if new_source and new_source != existing_source:
                        if isinstance(existing_source, list):
                            if new_source not in existing_source:
                                existing_source.append(new_source)
                        else:
                            existing_rel["source"] = [existing_source, new_source] if existing_source else new_source
                    break
    
    return existing


def _normalize_name(name: str) -> str:
    """Normalize entity name for deduplication."""
    if not name:
        return ""
    # Lowercase, strip whitespace
    normalized = name.lower().strip()
    return normalized


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _names_similar(word1: str, word2: str) -> bool:
    """
    Check if two name parts are similar enough to be the same name.
    Handles OCR errors and spelling variations.
    """
    if not word1 or not word2:
        return False
    if word1 == word2:
        return True
    
    # Must start with same letter (most name variations preserve first letter)
    if word1[0] != word2[0]:
        return False
    
    # Calculate edit distance
    distance = _levenshtein_distance(word1, word2)
    max_len = max(len(word1), len(word2))
    
    # Allow more edits for longer words
    # For short words (<=5 chars): allow 1-2 edits
    # For medium words (6-8 chars): allow 2-3 edits
    # For longer words: allow ~30% difference
    if max_len <= 5:
        return distance <= 2
    elif max_len <= 8:
        return distance <= 3
    else:
        return distance / max_len <= 0.3


def _names_match(name1: str, name2: str) -> bool:
    """Check if two names refer to the same entity."""
    n1 = _normalize_name(name1)
    n2 = _normalize_name(name2)
    
    if not n1 or not n2:
        return False
    
    # Exact match
    if n1 == n2:
        return True
    
    words1 = n1.split()
    words2 = n2.split()
    
    # One contains the other (e.g., "Amanda" in "Amanda Lynn Plasse")
    if n1 in n2 or n2 in n1:
        # Only match if it's a word boundary (not partial word)
        set1 = set(words1)
        set2 = set(words2)
        # If all words of one are in the other, it's a match
        if set1.issubset(set2) or set2.issubset(set1):
            return True
    
    # Check for spelling variations (e.g., "Plasse" vs "Plosh" vs "Ploss")
    # If first name matches exactly and last names are similar
    if len(words1) >= 1 and len(words2) >= 1:
        # Check if any word matches exactly
        common_words = set(words1) & set(words2)
        if common_words:
            # If we have a common word (like a first name), check if other words are similar
            remaining1 = [w for w in words1 if w not in common_words]
            remaining2 = [w for w in words2 if w not in common_words]
            
            # If all remaining words are similar, consider it a match
            if remaining1 and remaining2:
                for w1 in remaining1:
                    for w2 in remaining2:
                        if _names_similar(w1, w2):
                            return True
            elif not remaining1 or not remaining2:
                # One name is fully contained in the common words
                return True
        
        # Also check if first words match and we're just dealing with middle/last name variations
        # e.g., "Amanda Plasse" vs "Amanda Plosh"
        if words1[0] == words2[0] and len(words1) >= 2 and len(words2) >= 2:
            # First names match exactly, check if last names are similar
            if _names_similar(words1[-1], words2[-1]):
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
    Detect contradictions across claims from different documents.
    
    Two phases:
    1. Heuristic: Flag cross-document claims about same subject (for awareness, low priority)
    2. LLM: Find real contradictions where statements cannot both be true (high priority)
    """
    
    conflicts = []
    claims = all_data.get("claims", [])
    
    if len(claims) < 2:
        return conflicts
    
    # Phase 1: Heuristic - only flag CROSS-DOCUMENT claims (from 2+ different sources)
    # This is for awareness, not necessarily contradictions
    by_subject = defaultdict(list)
    for claim in claims:
        subject = claim.get("subject", "").lower().strip()
        if subject:
            by_subject[subject].append(claim)
    
    for subject, subject_claims in by_subject.items():
        sources = set(c.get("source", "") for c in subject_claims)
        
        # Only flag if claims come from MULTIPLE documents (cross-document)
        if len(sources) > 1:
            unique_claim_texts = set(c.get("claim", "").lower() for c in subject_claims)
            
            # Only flag if the claims are actually different
            if len(unique_claim_texts) > 1:
                conflicts.append({
                    "subject": subject,
                    "type": "cross_document_claims",
                    "claims": subject_claims,
                    "sources": list(sources),
                    "description": f"'{subject.title()}' is discussed in multiple documents with different claims"
                })
    
    # Phase 2: LLM - find REAL contradictions (statements that cannot both be true)
    if client and model and claims:
        llm_conflicts = _detect_conflicts_with_llm(client, model, claims)
        conflicts.extend(llm_conflicts)
    
    return conflicts


def _detect_conflicts_with_llm(client: OpenAI, model: str, claims: list) -> list:
    """Use LLM to detect contradictions AND inconsistencies in statements."""
    
    if len(claims) < 2:
        return []
    
    # Limit claims to avoid token limits
    claims_sample = claims[:50]
    
    prompt = """You are an investigative analyst reviewing witness statements. Flag ONLY genuinely problematic contradictions or inconsistencies.

## CONTRADICTIONS (High Priority) 🔴
Statements about the SAME SPECIFIC EVENT that CANNOT BOTH BE TRUE:
- Same event, different times: "I left at 8am" vs "I left at 10am"
- Same moment, different locations: "I was at work at 3pm" vs "He was at her house at 3pm"
- Statement vs physical evidence: "I never touched it" vs "Fingerprints match"
- Conflicting direct observations: "Wearing red shirt" vs "Wearing blue shirt"

## INCONSISTENCIES (Medium Priority) 🟡
SAME PERSON describing the SAME EVENT differently across statements:
- Changed timeline: "I arrived at 8" in Interview 1, "I arrived at 9" in Interview 2
- Added/removed people: First account mentions X present, second omits X entirely
- Sequence change: "I called then drove over" vs later saying "I drove then called"
- Key fact change: First statement says X happened, later denies or changes X

## ABSOLUTELY DO NOT FLAG ❌
These are NOT inconsistencies - do not flag them:
- Different emotions (suicidal AND caring about someone = both can be true)
- Different topics (one claim about drugs, another about relationships)
- Same event from different people's perspectives (they saw different things)
- Complementary details (one adds info the other doesn't have)
- General statements vs specific statements (unless they actually conflict)
- Things that happened at different times
- Different aspects of someone's character or mental state

CRITICAL: An inconsistency requires the SAME TOPIC being described DIFFERENTLY by the SAME PERSON or about the SAME SPECIFIC MOMENT.

CLAIMS TO ANALYZE:
"""
    for i, claim in enumerate(claims_sample):
        prompt += f"\n{i+1}. [{claim.get('source', 'unknown')}] {claim.get('subject', 'unknown')}: {claim.get('claim', '')}"
        if claim.get('quote'):
            quote_text = claim.get('quote')
            if isinstance(quote_text, list):
                quote_text = quote_text[0] if quote_text else ""
            prompt += f' (Quote: "{str(quote_text)[:100]}...")'
    
    prompt += """

Before flagging anything, ask yourself:
1. Are these claims about the EXACT SAME event/moment?
2. Is it the SAME PERSON giving different accounts, OR two people describing the SAME moment differently?
3. Do these claims actually conflict, or are they just different topics?

If you can't answer YES to #1 AND #2, do NOT flag it.

Return ONLY valid JSON array:
[
    {
        "claim_indices": [index1, index2],
        "type": "contradiction|inconsistency",
        "severity": "high|medium",
        "category": "timeline|location|sequence|story_change",
        "same_event": "what specific event/moment both claims are about",
        "claim1_says": "what the first claim asserts about that event",
        "claim2_says": "how the second claim differs about THE SAME event",
        "investigative_note": "why an investigator should care about this discrepancy"
    }
]

- "high" severity = cannot both be true (real contradiction)
- "medium" severity = same event described differently (suspicious inconsistency)

Return empty array [] if no genuine conflicts found. Most claim sets will have zero conflicts.
Be extremely conservative - false positives waste investigator time.
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
        
        # Enrich with actual claim data and validate
        enriched = []
        for conflict in llm_conflicts:
            indices = conflict.get("claim_indices", [])
            conflict["claims"] = [claims_sample[i] for i in indices if i < len(claims_sample)]
            conflict["sources"] = list(set(c.get("source", "") for c in conflict["claims"]))
            
            # VALIDATION: For inconsistencies, both claims must be about the same person
            # Different people can't be "inconsistent" with each other - they're just different accounts
            if conflict.get("type") == "inconsistency" and len(conflict["claims"]) >= 2:
                subjects = [c.get("subject", "").lower().strip() for c in conflict["claims"]]
                # Normalize names - check if they refer to the same person
                # (e.g., "Dennis" and "Dennis Rosa Roman" should match)
                subject_words = [set(s.split()) for s in subjects]
                
                # Check if subjects share any significant words (name parts)
                if len(subject_words) >= 2:
                    common_words = subject_words[0] & subject_words[1]
                    # Filter out common non-name words
                    non_name_words = {"the", "a", "an", "unknown", "person", "subject"}
                    meaningful_common = common_words - non_name_words
                    
                    if not meaningful_common:
                        # Different people - skip this false positive
                        print(f"Rejecting false positive: '{subjects[0]}' vs '{subjects[1]}' are different people")
                        continue
            
            enriched.append(conflict)
        
        return enriched
        
    except Exception as e:
        print(f"LLM conflict detection error: {e}")
        return []


def analyze_investigative_notes(client: OpenAI, model: str, all_data: dict) -> list:
    """
    Analyze extracted data across all documents to identify factual observations
    that warrant investigative follow-up.
    
    IMPORTANT: This function only identifies OBJECTIVE, VERIFIABLE facts.
    It does NOT make psychological judgments or assess credibility subjectively.
    """
    
    claims = all_data.get("claims", [])
    events = all_data.get("events", [])
    entities = all_data.get("entities", [])
    
    if len(claims) < 2:
        return []
    
    # Get people for context - build as simple dicts
    people = [e for e in entities if e.get("type") == "Person"]
    people_summary = [{"name": p.get("name"), "description": p.get("description")} for p in people[:20]]
    
    # Build JSON strings outside f-string to avoid escaping issues
    people_json = json.dumps(people_summary, indent=2)
    claims_json = json.dumps(claims[:40], indent=2)
    events_json = json.dumps(events[:20], indent=2)
    
    prompt = f"""Analyze this extracted evidence and identify FACTUAL OBSERVATIONS that investigators should be aware of.

CRITICAL RULES - READ CAREFULLY:
1. ONLY flag things that are OBJECTIVELY VERIFIABLE from the documents
2. You MUST cite specific quotes as evidence for every observation
3. DO NOT make psychological judgments (nervous, evasive, suspicious, rehearsed, lying)
4. DO NOT assess credibility, character, or intent
5. DO NOT speculate about what someone was thinking or feeling
6. Present FACTS ONLY, not interpretations
7. If you cannot cite a specific quote, do not include the observation

ACCEPTABLE observations (factual):
✓ "In Document A, X said they left at 8am. In Document B, X said they left at 10am." (statement_change)
✓ "X states they were at the victim's location on the morning of the incident." (proximity_to_incident)  
✓ "X claims Y was present, but no statement from Y exists in the documents." (unverified_claim)
✓ "X denies being in the kitchen, but fingerprints matching X were found there." (physical_evidence_conflict)
✓ "There is a 3-hour gap in X's account between 2pm and 5pm." (timeline_gap)
✓ "Documents show X had a financial dispute with the victim over $500." (financial_connection)

NOT ACCEPTABLE (subjective/inferential):
✗ "X's alibi seems rehearsed or overly detailed"
✗ "X appears evasive or nervous"
✗ "X's behavior is suspicious"
✗ "X may be lying"
✗ "X seems credible/not credible"

PEOPLE IN THE CASE:
{people_json}

CLAIMS/STATEMENTS MADE:
{claims_json}

TIMELINE OF EVENTS:
{events_json}

Return ONLY valid JSON array. Each observation MUST have specific quotes as evidence:
[
    {{
        "type": "statement_change|timeline_gap|proximity_to_incident|physical_evidence_conflict|unverified_claim|financial_connection",
        "subject": "person or topic this concerns",
        "observation": "neutral, factual description of what the documents show",
        "evidence": [
            {{"document": "filename", "quote": "exact quote from document"}},
            {{"document": "filename", "quote": "second quote if comparing statements"}}
        ],
        "follow_up_question": "factual question this raises (not accusatory)"
    }}
]

If no significant factual observations found, return empty array: []
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
        
        notes = json.loads(response_text)
        
        # Validate each note has required evidence
        validated_notes = []
        for note in notes:
            if note.get("evidence") and len(note["evidence"]) > 0:
                # Ensure evidence has quotes
                has_quote = any(e.get("quote") for e in note["evidence"])
                if has_quote:
                    validated_notes.append(note)
        
        return validated_notes
        
    except Exception as e:
        print(f"Investigative notes analysis error: {e}")
        return []


def remove_document_extraction(doc_name: str) -> dict:
    """Remove all extracted data for a specific document."""
    all_data = load_extracted()
    
    # Remove document from list
    if doc_name in all_data.get("documents", []):
        all_data["documents"].remove(doc_name)
    
    # Helper to check if source matches (handles both string and list sources)
    def source_matches(item_source, target_doc):
        if isinstance(item_source, list):
            return target_doc in item_source
        return item_source == target_doc
    
    # Filter out entities from this document
    all_data["entities"] = [e for e in all_data.get("entities", []) 
                           if not source_matches(e.get("source"), doc_name)]
    
    # Filter out claims from this document
    all_data["claims"] = [c for c in all_data.get("claims", []) 
                         if not source_matches(c.get("source"), doc_name)]
    
    # Filter out events from this document
    all_data["events"] = [e for e in all_data.get("events", []) 
                         if not source_matches(e.get("source"), doc_name)]
    
    # Filter out relationships from this document
    all_data["relationships"] = [r for r in all_data.get("relationships", []) 
                                 if not source_matches(r.get("source"), doc_name)]
    
    # Filter out key facts from this document
    all_data["key_facts"] = [f for f in all_data.get("key_facts", []) 
                            if isinstance(f, dict) and not source_matches(f.get("source"), doc_name)]
    
    # Filter out benefit indicators from this document
    all_data["benefit_indicators"] = [b for b in all_data.get("benefit_indicators", []) 
                                      if not source_matches(b.get("source"), doc_name)]
    
    # Clear conflicts and investigative notes (will be recalculated)
    all_data["conflicts"] = []
    all_data["investigative_notes"] = []
    
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
        "relationships": len(all_data.get("relationships", [])),
        "investigative_notes": len(all_data.get("investigative_notes", [])),
        "conflicts": len(all_data.get("conflicts", [])),
        "key_facts": len(all_data.get("key_facts", [])),
        "benefit_indicators": len(all_data.get("benefit_indicators", []))
    }

