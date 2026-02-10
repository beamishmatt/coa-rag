"""
Query router module for directing queries to the appropriate handler.

Routing strategy:
- GRAPH: Entity lookups, comprehensive lists, conflicts, timelines - use preprocessed knowledge graph
- VECTOR: Deep analysis, complex reasoning, multi-hop questions - use CoA + file_search
- HYBRID: Start with graph, augment with vector if needed (future)
- DEFLECT: Guilt/culpability determinations - gracefully decline to answer
- MOTIVE: Financial/benefit queries - answer factually with anti-inference framing
"""

import re
from typing import Tuple, Optional, List
from openai import OpenAI
from .extract import load_extracted, get_extraction_summary


# ============================================================================
# MOTIVE/BENEFIT QUERY DETECTION AND FRAMING
# ============================================================================

# Patterns for queries asking about who benefits/gains from the crime
MOTIVE_QUERY_PATTERNS = [
    r'\bwho\s+benefit',
    r'\bwho\s+gains?\b',
    r'\bcui\s+bono\b',
    r'\bwho\s+profits?\b',
    r'\bwho\s+had\s+(?:a\s+)?motive',
    r'\bwho\s+stood\s+to\s+gain\b',
    r'\bwho\s+stands?\s+to\s+gain\b',
    r'\bwho\s+would\s+benefit\b',
    r'\bwho\s+benefits?\s+from\b',
    r'\bwho\s+gained\s+from\b',
    r'\bmotive\s+(?:for|behind)\b',
    r'\bfinancial\s+motive\b',
    r'\bbeneficiary\b.*\b(?:who|list)\b',
    r'\bwho\s+inherits?\b',
    r'\bwho\s+gets?\s+(?:the\s+)?(?:money|insurance|estate|inheritance)\b',
]

MOTIVE_DISCLAIMER = """
---

⚠️ **Important Note for Investigators**

The financial relationships listed above are **documented facts from the case files**, not conclusions about guilt or involvement. Having a potential financial interest:

- Does NOT indicate guilt or culpability
- Does NOT establish motive on its own
- Must be evaluated alongside all other evidence
- Should be corroborated through independent investigation

This system reports what the documents contain — conclusions are yours to draw as the investigator.
"""


def is_motive_query(question: str) -> bool:
    """
    Detect if a question is asking about who benefits, gains, or has motive.
    These queries are answered factually with documented financial relationships.
    
    Args:
        question: The user's question
        
    Returns:
        True if the query is asking about benefit/motive
    """
    question_lower = question.lower().strip()
    
    for pattern in MOTIVE_QUERY_PATTERNS:
        if re.search(pattern, question_lower):
            return True
    
    return False


def identify_victim(data: dict) -> Optional[dict]:
    """
    Identify the victim/deceased person from extracted entities.
    
    Looks for entities marked as deceased, victim, killed, murdered, etc.
    Returns the entity dict if found, None otherwise.
    """
    entities = data.get("entities", [])
    
    # Keywords that indicate victim/deceased status
    victim_keywords = [
        "victim", "deceased", "killed", "murdered", "dead", "death",
        "homicide", "found dead", "body found", "body discovered"
    ]
    
    candidates = []
    
    for entity in entities:
        if entity.get("type", "").lower() != "person":
            continue
            
        description = entity.get("description", "").lower()
        mentions = " ".join(str(m).lower() for m in entity.get("mentions", []))
        
        # Check if this entity is marked as victim/deceased
        for keyword in victim_keywords:
            if keyword in description or keyword in mentions:
                candidates.append(entity)
                break
    
    # If we found candidates, return the one with most evidence
    if candidates:
        # Prefer the one with "victim" in description
        for c in candidates:
            if "victim" in c.get("description", "").lower():
                return c
        return candidates[0]
    
    # Fallback: Look for death events and find the person involved
    events = data.get("events", [])
    for event in events:
        desc = event.get("description", "").lower()
        if any(kw in desc for kw in ["killed", "murdered", "death", "died", "found dead", "body"]):
            people = event.get("people_involved", [])
            if people:
                # Find the entity for this person
                for person_name in people:
                    for entity in entities:
                        if entity.get("type", "").lower() == "person":
                            if _normalize_for_matching(person_name) in _normalize_for_matching(entity.get("name", "")):
                                return entity
    
    return None


def get_people_connected_to_victim(data: dict, victim_name: str) -> List[dict]:
    """
    Get all people who have a documented relationship with the victim.
    
    Returns list of dicts with person info and their relationship to victim.
    """
    relationships = data.get("relationships", [])
    entities = data.get("entities", [])
    claims = data.get("claims", [])
    
    victim_normalized = _normalize_for_matching(victim_name)
    victim_words = set(victim_normalized.split())
    
    connected_people = {}
    
    # Find relationships involving the victim
    for rel in relationships:
        p1 = rel.get("person1", "")
        p2 = rel.get("person2", "")
        p1_norm = _normalize_for_matching(p1)
        p2_norm = _normalize_for_matching(p2)
        
        other_person = None
        
        # Check if victim is person1 or person2
        if victim_normalized in p1_norm or any(w in p1_norm.split() for w in victim_words):
            other_person = p2
        elif victim_normalized in p2_norm or any(w in p2_norm.split() for w in victim_words):
            other_person = p1
        
        if other_person:
            other_norm = _normalize_for_matching(other_person)
            if other_norm not in connected_people:
                connected_people[other_norm] = {
                    "name": other_person,
                    "relationships": [],
                    "claims": [],
                    "entity_info": None
                }
            connected_people[other_norm]["relationships"].append(rel)
    
    # Find entity info for connected people
    for person_key in connected_people:
        for entity in entities:
            if entity.get("type", "").lower() == "person":
                entity_norm = _normalize_for_matching(entity.get("name", ""))
                if person_key in entity_norm or entity_norm in person_key:
                    connected_people[person_key]["entity_info"] = entity
                    break
    
    # Find claims about connected people
    for claim in claims:
        subject = claim.get("subject", "")
        subject_norm = _normalize_for_matching(subject)
        
        for person_key in connected_people:
            if person_key in subject_norm or subject_norm in person_key:
                connected_people[person_key]["claims"].append(claim)
    
    return list(connected_people.values())


# ============================================================================
# SPECULATIVE QUERY DETECTION AND DEFLECTION
# ============================================================================

# Patterns for speculative questions that ask about motives, reasons, or who benefits
SPECULATIVE_QUERY_PATTERNS = [
    # "Who benefits" variations
    r'\bwho\s+benefit',
    r'\bwho\s+gains?\b',
    r'\bcui\s+bono\b',
    r'\bwho\s+profits?\b',
    r'\bwho\s+had\s+(?:a\s+)?motive',
    r'\bwho\s+stood\s+to\s+gain\b',
    r'\bwho\s+stands?\s+to\s+gain\b',
    r'\bwho\s+would\s+benefit\b',
    r'\bwho\s+benefits?\s+from\b',
    r'\bwho\s+gained\s+from\b',
    r'\bbeneficiary\b.*\b(?:who|list)\b',
    # "Why was X killed/murdered" variations
    r'\bwhy\s+was\s+\w+\s+killed\b',
    r'\bwhy\s+was\s+\w+\s+murdered\b',
    r'\bwhy\s+did\s+\w+\s+(?:die|get\s+killed|get\s+murdered)\b',
    r'\bwhy\s+(?:the\s+)?(?:murder|killing|death|crime)\b',
    r'\breason\s+(?:for|behind)\s+(?:the\s+)?(?:murder|killing|death|crime)\b',
    r'\bmotive\s+(?:for|behind)\s+(?:the\s+)?(?:murder|killing|death|crime)\b',
    r'\bwhat\s+(?:is|was)\s+the\s+motive\b',
    r'\bwhy\s+would\s+(?:someone|anyone)\s+(?:kill|murder|want\s+.+\s+dead)\b',
]

SPECULATIVE_DEFLECTION_RESPONSE = """## Query Outside System Scope

I'm designed to help investigators **find and organize factual evidence** from documents — not to speculate about motives or reasons behind crimes.

### Why I Can't Answer This

Questions like "Who benefits from the crime?" or "Why was [person] killed?" require:

- **Speculation about intent** — which is outside the scope of documentary evidence
- **Inference about motive** — which can lead to confirmation bias
- **Assumptions about relationships** — that may not be supported by facts

Even if financial or relational connections exist in the documents, **inferring motive from those connections is the investigator's job**, not mine.

### What I Can Help With Instead

Try asking factual questions:

- **Financial relationships**: "What insurance policies are mentioned?" / "Who is listed as a beneficiary?"
- **Conflicts and arguments**: "Were there any documented disputes involving [person]?"
- **Relationships**: "What is the relationship between [person A] and [person B]?"
- **Timeline**: "Who was the last person to see [victim] alive?"
- **Statements**: "What did [person] say about [topic]?"

### Why This Matters

Drawing conclusions about who benefited or why someone was killed from documentary evidence alone can:

- Introduce bias into the investigation
- Lead to tunnel vision on certain suspects
- Miss alternative explanations

My role is to help you **find facts** — the conclusions are yours to draw.

---

*Rephrase your question to ask about specific facts, statements, or documented relationships.*"""


def is_speculative_query(question: str) -> bool:
    """
    Detect if a question is asking speculative questions about motive,
    who benefits, or reasons behind crimes.
    
    These queries are deflected because they ask for inference/speculation
    rather than factual retrieval.
    
    Args:
        question: The user's question
        
    Returns:
        True if the query is speculative (should be deflected)
    """
    question_lower = question.lower().strip()
    
    for pattern in SPECULATIVE_QUERY_PATTERNS:
        if re.search(pattern, question_lower):
            return True
    
    # Simple phrase matching for common variations
    speculative_phrases = [
        "who benefits",
        "who benefited",
        "who would benefit",
        "who stands to gain",
        "who stood to gain",
        "why was she killed",
        "why was he killed",
        "why were they killed",
        "why the murder",
        "why the killing",
        "what was the motive",
        "what is the motive",
        "reason for the murder",
        "reason for the killing",
    ]
    
    for phrase in speculative_phrases:
        if phrase in question_lower:
            return True
    
    return False


def get_speculative_deflection_response() -> str:
    """
    Return the standard deflection response for speculative queries.
    """
    return SPECULATIVE_DEFLECTION_RESPONSE


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


# ============================================================================
# FOLLOW-UP QUESTION QUERY DETECTION
# ============================================================================

FOLLOW_UP_QUERY_PATTERNS = [
    r'\bfollow[\s-]?up\s+questions?\s+(?:for|about)\b',
    r'\bgenerate\s+.*\bquestions?\s+(?:for|about)\b',
    r'\bwhat\s+(?:questions?|should\s+(?:i|we))\s+ask\b',
    r'\binterview\s+questions?\s+(?:for|about)\b',
    r'\bquestions?\s+(?:to|i\s+should|we\s+should)\s+ask\b',
    r'\bwhat\s+(?:else\s+)?(?:should|can|could)\s+(?:i|we)\s+ask\b',
    r'\bprepare\s+questions?\s+(?:for|about)\b',
    r'\bquestions?\s+(?:for|to)\s+(?:the\s+)?(?:next\s+)?interview\b',
]

FOLLOW_UP_QUERY_PHRASES = [
    "follow up questions",
    "follow-up questions",
    "followup questions",
    "questions to ask",
    "questions for",
    "interview questions",
    "what should i ask",
    "what should we ask",
    "what questions",
    "generate questions",
    "list of questions",
]


def is_follow_up_query(question: str) -> bool:
    """
    Detect if a question is asking for follow-up/interview questions for a person.
    
    Args:
        question: The user's question
        
    Returns:
        True if the query is asking for follow-up questions
    """
    question_lower = question.lower().strip()
    
    # Check regex patterns
    for pattern in FOLLOW_UP_QUERY_PATTERNS:
        if re.search(pattern, question_lower):
            return True
    
    # Check simple phrases
    for phrase in FOLLOW_UP_QUERY_PHRASES:
        if phrase in question_lower:
            return True
    
    return False


def extract_target_person_for_follow_up(question: str, entities: List[dict]) -> Optional[dict]:
    """
    Extract the target person from a follow-up question query.
    
    Args:
        question: The user's question (e.g., "generate follow-up questions for Dennis")
        entities: List of entity dicts from extracted data
        
    Returns:
        The matching entity dict if found, None otherwise
    """
    # Filter to person entities only
    person_entities = [e for e in entities if e.get("type", "").lower() == "person"]
    
    if not person_entities:
        return None
    
    # First, try extracting capitalized names from the question
    potential_names = _extract_potential_names(question)
    
    if potential_names:
        matches = _find_matching_entities(potential_names, person_entities)
        if matches:
            return matches[0]
    
    # Fallback: Search for any person entity name mentioned in the question (case-insensitive)
    # This handles cases like "questions for dennis" where dennis is lowercase
    question_lower = question.lower()
    
    best_match = None
    best_match_score = 0
    
    for entity in person_entities:
        entity_name = entity.get("name", "")
        entity_name_lower = entity_name.lower()
        
        # Check if any part of the entity name appears in the question
        name_parts = entity_name_lower.split()
        
        for part in name_parts:
            if len(part) >= 3 and part in question_lower:  # Minimum 3 chars to avoid false positives
                # Score by how much of the name matches
                score = len(part)
                if score > best_match_score:
                    best_match = entity
                    best_match_score = score
        
        # Also check full name match
        if entity_name_lower in question_lower:
            # Full name match is highest priority
            return entity
    
    return best_match


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
    
    # Use comprehensive benefit analysis framing for motive queries
    if category == "motive":
        system_prompt = """You are an investigative analyst presenting a comprehensive benefit analysis based on documented evidence. Your role is to systematically analyze who might have benefited from the victim's death across THREE categories, while maintaining objectivity.

ANALYSIS STRUCTURE (use this exact format):

## Benefit Analysis for [Victim Name]

### Financial Benefit
Analyze documented evidence of who might gain financially:
- Insurance beneficiaries
- Inheritance/estate beneficiaries  
- People who owed money TO the victim (debts cleared by death)
- Business partners who gain control/assets

For each person with potential financial benefit:
- State the documented relationship
- Quote the supporting evidence
- Note the source

If no financial benefit evidence: Say "No documented financial beneficiaries or debt relationships found in the documents you provided."

### Practical Benefit
Analyze documented evidence of who might gain practically:
- People the victim had conflicts with (obstacle removed)
- People who might gain power/control
- People whose secrets the victim knew (secrets protected)
- Witnesses to something the victim knew about

For each person with potential practical benefit:
- State the documented conflict or situation
- Quote the supporting evidence
- Note the source

If no practical benefit evidence: Say "No documented conflicts or strategic relationships found in the documents you provided."

### Emotional/Relational Benefit
Analyze documented evidence of emotional dynamics:
- Romantic jealousy (love triangles, affairs)
- Revenge for prior grievances
- Custody/divorce disputes
- Long-standing personal conflicts

For each person with potential emotional benefit:
- State the documented relationship dynamic
- Quote the supporting evidence (statements about anger, jealousy, conflicts)
- Note the source

If no emotional benefit evidence: Say "No documented emotional conflicts found in the documents you provided."

CRITICAL RULES:

1. **Evidence-Based Only:** Every benefit claim MUST cite a specific quote or document
2. **No Speculation:** Only include benefits supported by documentary evidence
3. **Objective Framing:** 
   - SAY: "Documents show X and victim had a romantic conflict..."
   - NOT: "X was jealous and had motive..."
4. **Present All Connected People:** Even if their benefit is unclear, note their documented relationship to the victim
5. **Note Gaps:** If a category has no evidence in the documents you provided, explicitly state this - it's useful information
6. **INTERVIEW STATEMENTS ARE NOT FACTS:** When citing interview content, use hedging language:
   - "According to [Person]'s statement..." NOT "The fact is..."
   - "[Person] claimed..." NOT "[Person] did..."
   - People lie, misremember, and protect themselves - their statements are claims, not verified truth

The disclaimer at the end is pre-included - do not add another one.

Remember: Your analysis helps investigators see the full picture of relationships. Present facts systematically without drawing conclusions about guilt."""

    # Use formal law enforcement narrative style for case summaries
    elif category == "summary":
        system_prompt = """You are a law enforcement analyst preparing a case briefing. Write in formal narrative prose suitable for professional investigative review.

WRITING STYLE:
- Write in flowing prose paragraphs, not bullet lists
- Use direct, declarative sentences
- Maintain a clinical, objective tone throughout
- Use police report-style phrasing ("On [date], the body of [victim] was discovered...", "The victim was identified as...")
- Attribute all statements formally ("According to [witness]...", "[Person] stated during interview that...")

STRUCTURE YOUR BRIEFING:
1. **Incident Summary** - Date, location, victim identification, nature of incident
2. **Victim Background** - Known information about the victim prior to the incident
3. **Sequence of Events** - Chronological reconstruction based on witness accounts and evidence
4. **Witness Accounts** - Summary of statements from interviewed individuals
5. **Physical Evidence** - Evidence collected and forensic findings
6. **Outstanding Issues** - Unresolved questions, conflicting accounts, investigative gaps

Use ## for section headers. Within sections, write prose paragraphs.

PROHIBITED LANGUAGE - DO NOT USE:
- Dramatic openers ("tragedy struck", "tranquility shattered", "harrowing incident", "brutal")
- Editorializing adjectives ("compelling", "intriguing", "suspicious", "notable")
- Magazine-style phrases ("raises questions", "adds complexity", "web of connections", "murky timeline")
- Narrative tension language ("interestingly", "notably", "curiously", "significantly")
- Speculative phrasing ("may have", "could suggest", "might indicate", "begs the question")
- Emotional language ("fateful day", "untimely demise", "grim discovery", "demise")
- META-COMMENTARY about the system or extraction process - NEVER mention:
  - "identified X entities", "extracted Y claims", "documented Z events"
  - "significant progress has been made", "the investigation has identified"
  - Any reference to internal data counts, extraction, or system processes
- VAGUE INVESTIGATIVE SUGGESTIONS like:
  - "need to clarify relationships", "ascertain additional evidence"
  - "requires further investigation", "warrants additional scrutiny"
  - Present facts only - do not tell investigators what to do next

CORRECT STYLE EXAMPLES:
- WRONG: "The tranquility of Chicopee was shattered by the brutal homicide..."
- RIGHT: "On August 26, 2011, the body of Amanda Lynn Plasse, age 20, was discovered in Chicopee."

- WRONG: "This raises intriguing questions about his involvement..."
- RIGHT: "Seth Green's fingerprints were recovered from the crime scene. Green stated he was working at a job site on the day of the incident."

- WRONG: "The investigation took a critical turn when forensic evidence emerged..."
- RIGHT: "Crime scene analysis yielded fingerprint evidence. Fingerprints matching Seth Green were identified at the scene."

ATTRIBUTION REQUIREMENTS:
- State what each person said, not what happened: "[Person] stated..." not "[Person] was..."
- Distinguish between verified facts and claims made by individuals
- When accounts conflict, present both versions with their sources

INTERVIEW STATEMENTS ARE NOT FACTS (CRITICAL):
Just because someone said something in an interview does NOT mean it happened. People lie, misremember, have biases, and protect themselves or others.
- ALWAYS hedge interview content: "[Person] stated..." or "According to [Person]'s account..."
- "X stated they were at the store" NOT "X was at the store"
- "X claimed to have seen Y" NOT "X saw Y"
- When someone describes their own alibi, explicitly note this is THEIR CLAIM
- Only physical evidence (forensics, photos) can be stated as direct fact

ANTI-HALLUCINATION RULES:
- Include ONLY facts from the extracted data
- Never invent or embellish details
- If information is not in the data, do not include it

INVESTIGATIVE OBJECTIVITY:
- NEVER label anyone as a "suspect" unless documents explicitly use that term
- Use neutral language: "interviewed individual", "person of interest" only if documented
- Present facts without prejudging guilt, innocence, or involvement
- Do not draw conclusions about culpability

LOGICAL CONSISTENCY:
- Never combine contradictory facts into a single statement
- If person X discovered the body, X cannot be "the last to see them alive"
- Present conflicting accounts separately with attribution

TEMPORAL RELATIONSHIP RULES - CRITICAL:
- NEVER invert "before" and "after" relationships when rewriting facts
- If the data says "X happened after Y saw them" → write "Y saw them before X happened"
- If someone is deceased, any past interaction MUST be stated as "before their death"
- Verify temporal logic before writing: Can this sequence actually happen in this order?
- A living person cannot see/interact with a dead person (unless discovering the body)
- WRONG: "Dennis last saw Amanda after she had been killed" (impossible)
- RIGHT: "Dennis's last contact with Amanda occurred before her death"
- When uncertain about temporal ordering, keep facts separate rather than combining them"""

    else:
        # Use structured bullet-point format for all other query types
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

DO NOT INCLUDE (unless specifically asked for a case summary):
- NO "Incident Overview" section
- NO "Key Individuals" section
- NO general case background or victim information unless directly relevant to the specific question
- Answer ONLY what was asked - do not pad with case summary information

EXACT QUOTES ARE MANDATORY:
- When the extracted data contains direct quotes (text in > blockquotes), you MUST include them
- NEVER paraphrase or summarize quotes - preserve the exact wording
- Format quotes using > blockquote syntax
- Always attribute quotes to their source document
- Quotes are critical evidence - do not drop them from your response

REASONING REQUIREMENTS:
- After presenting findings, include a brief "Reasoning" section
- Explain HOW you arrived at your conclusions based on the evidence
- Describe what evidence you found most relevant and why
- If you connected multiple pieces of information, explain those connections
- If you made any inferences, explicitly state them and justify with evidence

DEDUCTIVE REASONING (REQUIRED):
You MUST perform evidence-constrained deduction when partial information can be cross-referenced.

When evidence provides partial identifying information (e.g., "name starts with D", physical description, location access):
1. STATE THE CONSTRAINT: What partial information do we have?
2. ENUMERATE POSSIBILITIES: List ALL entities in the data that match the constraint
3. APPLY LOGICAL ELIMINATION: For each possibility, evaluate likelihood based on documented context
4. STATE CONCLUSION: "Based on this evidence, [Name] is the most likely candidate" or "This narrows to [names]"

Example - CORRECT approach:
> The name on the whiteboard "begins with a D" (Jesse Columbe). Cross-referencing all names:
> - Dennis Rosa Roman - delivered marijuana to Amanda, documented at her apartment
> - Dorcas Ortiz - Dennis's mother, no documented presence at apartment
> Dennis is the only "D" name with documented connection to the break-in context.

This is NOT speculation - it's evidence-constrained deduction using only facts from the documents.

CRITICAL ANTI-HALLUCINATION RULES:
- ONLY include information that appears in the extracted data provided
- If someone asks about a person/entity NOT in the data, say "No information found about [name] in the documents you provided"
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

INTERVIEW STATEMENTS ARE NOT FACTS (CRITICAL):
Just because someone said something in an interview does NOT mean it happened. People lie, misremember, have biases, and protect themselves or others.

- ALWAYS hedge interview content: "[Person] stated..." or "According to [Person]'s account..." - NEVER present as fact
- "X stated they were at the store" NOT "X was at the store"
- "X claimed to have seen Y" NOT "X saw Y"
- "According to X, they gave Y $50" NOT "X gave Y $50"
- When someone describes their own alibi or whereabouts, explicitly note this is THEIR CLAIM, not verified fact
- If multiple people's accounts differ, present BOTH as claims without favoring either
- Only physical evidence and official records (forensics, phone records, surveillance) can be stated more directly
- Add "(unverified)" or "(according to their interview)" when summarizing interview content

PROHIBITED META-COMMENTARY:
- NEVER mention internal system processes: "identified X entities", "extracted Y claims", "documented Z events"
- NEVER use progress language: "significant progress has been made", "the investigation has identified"
- NEVER give vague investigative suggestions: "need to clarify relationships", "ascertain additional evidence", "requires further investigation", "warrants additional scrutiny"
- Do NOT use euphemisms like "demise" - use direct language like "death" or "killed"
- Present facts from documents only - do not tell investigators what to do next

GUILT DETERMINATION - WHEN TO REFUSE:
- ONLY refuse if the question explicitly asks you to determine culpability: "who is guilty", "who did it", "who committed the crime", "who is the perpetrator/killer"
- If you must refuse, respond with: "I cannot determine guilt or culpability. I can help you find specific facts, statements, or evidence in the documents. Please rephrase your question."

IMPORTANT - These are NOT guilt determination queries (ANSWER NORMALLY):
- "Who was the last person to see [person] alive" → Factual timeline question - ANSWER IT
- "Who found the body" → Factual question - ANSWER IT
- "Who was with [person] on [date]" → Factual question - ANSWER IT
- "Who had access to [location]" → Factual question - ANSWER IT
- Questions about who saw whom, when, and where are FACTUAL questions that MUST be answered

LOGICAL CONSISTENCY - CRITICAL:
- NEVER combine statements that create logical impossibilities
- Before merging facts into a single sentence, verify they can logically coexist:
  - If person X "found the body" or "found them dead" → X was NOT "the last to see them alive"
  - If event A happened "before" event B → A cannot also happen "after" B  
  - If someone was "alive" at time T → they cannot be "dead" at time T
  - "Discovered the body" and "last person to see them alive" are MUTUALLY EXCLUSIVE
- When facts seem contradictory or incompatible, present them SEPARATELY with their sources
- Say: "According to Source A, X happened. According to Source B, Y happened." — do NOT merge into one statement
- If unsure whether facts are compatible, keep them separate rather than combining them
- NEVER write sentences that contain internal contradictions

TEMPORAL RELATIONSHIP RULES - CRITICAL:
- NEVER invert "before" and "after" relationships when rewriting facts
- If the data says "X happened after Y saw them" → write "Y saw them before X happened"
- If someone is deceased, any past interaction MUST be stated as "before their death"
- Verify temporal logic before writing: Can this sequence actually happen in this order?
- A living person cannot see/interact with a dead person (unless discovering the body)
- WRONG: "Dennis last saw Amanda after she had been killed" (impossible)
- RIGHT: "Dennis's last contact with Amanda occurred before her death"
- When uncertain about temporal ordering, keep facts separate rather than combining them

QUESTION-FOCUSED RESPONSE (CRITICAL):
- Read the user's question carefully and identify EXACTLY what they are asking
- Answer the SPECIFIC question in your FIRST SENTENCE - do not start with headers or overviews
- Do NOT give a general overview when a specific answer exists in the data
- Common question patterns and how to answer them:
  - "Who was the last person to see X alive" → FIRST SENTENCE: "[Name] was the last person documented to see/speak with X at [time]."
  - "Who was X with" → FIRST SENTENCE: "According to [source], X was with [names] at [time/place]."
  - "What did X say about Y" → FIRST SENTENCE: "[X] stated: '[quote]' (Source: [doc])"
  - "When did X happen" → FIRST SENTENCE: "[Event] occurred at [time/date] according to [source]."
  - "Where was X" → FIRST SENTENCE: "[Person] stated they were at [location]. (Source: [doc])"
  - "Who found the body" → FIRST SENTENCE: "[Name] found/discovered the body according to [source]."
- If the extracted data contains a direct answer to the question, lead with that answer
- Only include tangential information AFTER directly answering what was asked
- NEVER start with "Incident Overview", "Key Individuals", "Timeline" headers for specific questions"""

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
    
    # Common words that appear capitalized at sentence start but are NOT names
    non_name_words = {
        # Question words
        "what", "who", "where", "when", "why", "how", "which",
        # Command/action words often at start of queries
        "extract", "show", "list", "find", "get", "tell", "give", "display",
        "search", "look", "describe", "explain", "summarize", "identify",
        # Articles and common words
        "the", "a", "an", "all", "any", "some", "this", "that", "these", "those",
        # Verbs commonly starting sentences
        "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
        "should", "has", "have", "had",
        # Other common non-names
        "yes", "no", "please", "thanks", "okay", "evidence", "documents",
        "case", "information", "details", "facts", "names", "people", "person"
    }
    
    # Extract quoted strings
    quoted = re.findall(r'["\']([^"\']+)["\']', question)
    names.extend(quoted)
    
    # Extract capitalized word sequences (proper nouns)
    # Match sequences like "John Smith", "Detective Roman", "Amanda Lynn Plasse"
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
    
    # Filter out common non-name words
    for word in capitalized:
        if word.lower() not in non_name_words:
            names.append(word)
    
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


def _extract_letter_filter(question: str) -> Optional[str]:
    """
    Extract letter filter from questions like "names starting with D" or "beginning with A".
    
    Returns the uppercase letter if found, None otherwise.
    """
    # Strip all non-alphanumeric characters except spaces for simpler matching
    # This handles all quote variations (curly, straight, etc.)
    cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in question.lower())
    # Collapse multiple spaces
    cleaned = ' '.join(cleaned.split())
    
    # Patterns for letter filtering - simpler patterns after cleaning
    letter_patterns = [
        # "starting with D" or "beginning with D" or "starts with D"
        r'(?:start|starting|starts|begin|beginning|begins)\s+with\s+(?:a\s+)?(?:the\s+letter\s+)?([a-z])\b',
        # "names that start with D"
        r'names?\s+(?:that\s+)?(?:start|begin)s?\s+with\s+([a-z])\b',
        # "names with letter D"
        r'names?\s+with\s+(?:letter\s+)?([a-z])\b',
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, cleaned)
        if match:
            letter = match.group(1).upper()
            print(f"DEBUG: Extracted letter filter '{letter}' from question: {question[:50]}...")
            return letter
    
    return None


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
    
    # Check for letter-filter queries like "names starting with D" - these need knowledge graph
    letter_filter = _extract_letter_filter(question)
    print(f"_is_comprehensive_query: letter_filter = {letter_filter}")
    if letter_filter:
        print(f"_is_comprehensive_query: -> True (letter filter detected)")
        return True
    
    comprehensive_keywords = [
        "all ", "every ", "list ", "find all", "show all", "give me all",
        "inconsistencies", "contradictions", "conflicts", "discrepancies",
        "everyone", "everything", "everybody",
        # Summary queries - all variations should hit knowledge graph
        "summarize", "summary", "case summary", "give me a summary",
        "brief the case", "brief me", "case brief", "case overview",
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
        # Names queries
        "what names", "which names",
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
    1. Motive/benefit queries → EXHAUSTIVE (search extracted financial relationships)
    2. Comprehensive/list queries → EXHAUSTIVE (knowledge graph)
    3. Entity lookup queries → Check if entity exists in graph
       - Entity found → EXHAUSTIVE
       - Entity not found → SPECIFIC (search documents)
    4. Deep analysis queries → SPECIFIC (need document context)
    5. Default → SPECIFIC (CoA handles uncertainty well)
    """
    question_lower = question.lower().strip()
    print(f"CLASSIFY_QUERY: Processing '{question[:60]}...'")
    
    # 0. Follow-up question queries go to knowledge graph (need all evidence about a person)
    if is_follow_up_query(question):
        print(f"CLASSIFY_QUERY: -> EXHAUSTIVE (follow_up_query)")
        return "EXHAUSTIVE"
    
    # 1. Motive/benefit queries go to knowledge graph (financial relationships)
    if is_motive_query(question):
        print(f"CLASSIFY_QUERY: -> EXHAUSTIVE (motive_query)")
        return "EXHAUSTIVE"
    
    # 2. Comprehensive queries always go to knowledge graph
    is_comprehensive = _is_comprehensive_query(question)
    print(f"CLASSIFY_QUERY: _is_comprehensive_query returned {is_comprehensive}")
    if is_comprehensive:
        print(f"CLASSIFY_QUERY: -> EXHAUSTIVE (comprehensive_query)")
        return "EXHAUSTIVE"
    
    # 3. Check for entity lookup patterns
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
    
    Returns: "follow_up_questions", "conflicts", "entities", "events", "relationships", "investigative_notes", "motive", "summary", or "general"
    """
    question_lower = question.lower()
    
    # Check for follow-up question queries FIRST
    if is_follow_up_query(question):
        return "follow_up_questions"
    
    # Check for motive/benefit queries (before other patterns)
    if is_motive_query(question):
        return "motive"
    
    if any(kw in question_lower for kw in ["inconsisten", "contradict", "conflict", "discrepan"]):
        return "conflicts"
    
    # Check for relationship queries
    relationship_keywords = [
        "relationship", "relationships", "connected", "connection", 
        "know each other", "related to", "who knows", "associated with",
        "friends with", "family", "boyfriend", "girlfriend", "spouse",
        "coworker", "between", "related", "connections", "links", "linked"
    ]
    # More specific patterns for relationships
    if any(kw in question_lower for kw in relationship_keywords):
        # Make sure it's asking about relationships between people, not just using the word
        if any(p in question_lower for p in ["between", "who knows", "connected", "relationship", 
                                              "related", "entities", "people", "everyone",
                                              "all the", "connections", "links"]):
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
    
    # Follow-up questions handle their own LLM synthesis
    if category == "follow_up_questions":
        return _answer_follow_up_query(question, extracted_data, client, model)
    
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
    elif category == "motive":
        raw_response, success = _answer_motive_query(question, extracted_data)
    elif category == "summary":
        raw_response, success = _answer_summary_query(extracted_data)
    else:
        raw_response, success = _answer_general_exhaustive(question, extracted_data)
    
    # If client provided, synthesize a natural response
    # BUT skip synthesis for letter-filter queries since templates are already well-formatted
    # and LLM synthesis tends to corrupt the factual output
    if client and model and success:
        # Check if this is a letter-filter query - if so, skip synthesis
        if _extract_letter_filter(question):
            print(f"DEBUG: Skipping LLM synthesis for letter-filter query")
            return (raw_response, True)
        
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
            
            # Show source evidence with quotes
            response += "**Source Evidence:**\n\n"
            for claim in conflict.get("claims", []):
                source = claim.get("source", "Unknown source")
                claim_text = claim.get("claim", "No claim text")
                quote = claim.get("quote", "")
                
                response += f"- *{source}*: {claim_text}\n"
                if quote:
                    quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
                    if quote_text:
                        response += f"  > \"{str(quote_text)[:200]}{'...' if len(str(quote_text)) > 200 else ''}\"\n"
                response += "\n"
            response += "---\n\n"
    
    if not high_severity and not medium_severity:
        response += "_No contradictions or notable inconsistencies found in the statements in the documents you provided._\n\n"
    
    # Note: cross_doc_claims are filtered out but NOT displayed here
    # Being mentioned in multiple documents is not a contradiction - it's expected
    # Actual contradictions ACROSS documents are caught by high_severity/medium_severity
    
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
    
    # Check for letter-based filtering FIRST (e.g., "names starting with D")
    letter_filter = _extract_letter_filter(question)
    
    # First, check if this is a specific entity lookup (but not if we have a letter filter)
    if not letter_filter:
        potential_names = _extract_potential_names(question)
        if potential_names:
            matching_entities = _find_matching_entities(potential_names, entities)
            
            if matching_entities:
                return _answer_specific_entity_query(question, matching_entities, claims, data)
            else:
                # Names were mentioned but NOT found in documents - fail gracefully
                names_str = ", ".join(f'"{name}"' for name in potential_names)
                return (
                    f"## No Information Found in Provided Documents\n\n"
                    f"I searched the documents you provided but could not find any information about {names_str}.\n\n"
                    f"The following people ARE mentioned in the documents you provided:\n\n" +
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
    
    # Apply letter filter if present (e.g., "names starting with D")
    if letter_filter:
        print(f"DEBUG: Applying letter filter '{letter_filter}' to {len(filtered)} entities")
        before_count = len(filtered)
        filtered = [e for e in filtered if e.get("name", "").upper().startswith(letter_filter)]
        print(f"DEBUG: After filter: {len(filtered)} entities match (names: {[e.get('name') for e in filtered[:5]]})")
        
        if not filtered:
            # No entities match the letter filter - this shouldn't happen for common letters
            all_people = [e for e in entities if e.get("type", "").lower() == "person"]
            print(f"DEBUG: No matches found. All people: {[e.get('name') for e in all_people[:10]]}")
            return (
                f"## No Names Starting with \"{letter_filter}\" Found\n\n"
                f"I searched the documents but found no people whose names start with \"{letter_filter}\".\n\n"
                f"**People mentioned in the documents:**\n\n" +
                "\n".join(f"- **{e.get('name')}**" for e in all_people[:15]) +
                "\n\n*If you're looking for a partial name or nickname, try searching for the specific text.*",
                True
            )
    
    if not filtered:
        return (f"No {entity_type.lower()} were identified in the documents you provided.", True)
    
    # Deduplicate by name (case-insensitive)
    seen_names = set()
    unique_entities = []
    for e in filtered:
        name_lower = e.get("name", "").lower()
        if name_lower and name_lower not in seen_names:
            seen_names.add(name_lower)
            unique_entities.append(e)
    
    # Build response header
    if letter_filter:
        response = f"## Names Starting with \"{letter_filter}\"\n\n"
    else:
        response = f"## {entity_type} Mentioned in Documents\n\n"
    
    # Add disclaimer for role-based queries
    if role_based_query:
        response += "_**Note:** This system does not assume or assign investigative roles (suspect, witness, victim). The following is a list of all people mentioned in the documents. Any role designations shown are direct quotes from the source documents, not conclusions drawn by this system._\n\n"
    
    if letter_filter:
        response += f"Found **{len(unique_entities)}** name(s) starting with \"{letter_filter}\":\n\n"
    else:
        response += f"Found **{len(unique_entities)}** unique {entity_type.lower()}:\n\n"
    
    # Clean numbered list format
    for idx, entity in enumerate(unique_entities, 1):
        name = entity.get("name", "Unknown")
        desc = entity.get("description", "").replace("\n", " ").strip()
        source = entity.get("source", "Unknown source")
        
        # Format source
        if isinstance(source, list):
            source_str = ", ".join(source)
        else:
            source_str = source
        
        response += f"{idx}. **{name}**"
        if desc:
            response += f" — {desc}"
        response += f"\n   *Source: {source_str}*\n\n"
    
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
        
        # Find related relationships (who they were with, connections)
        relationships = data.get("relationships", [])
        related_relationships = []
        for rel in relationships:
            p1 = rel.get("person1", "").lower()
            p2 = rel.get("person2", "").lower()
            if entity_name_lower in p1 or entity_name_lower in p2 or \
               any(word in p1 or word in p2 for word in entity_words):
                related_relationships.append(rel)
        
        if related_relationships:
            response += "### Relationships\n\n"
            for rel in related_relationships[:10]:  # Limit to 10
                p1 = rel.get("person1", "")
                p2 = rel.get("person2", "")
                rel_type = rel.get("type", "")
                rel_desc = rel.get("description", "")
                quote = rel.get("quote", "")
                source = rel.get("source", "")
                
                response += f"**{p1}** — {rel_type} — **{p2}**\n\n"
                if rel_desc:
                    response += f"{rel_desc}\n\n"
                if quote:
                    response += f"> \"{quote}\"\n\n"
                response += f"*Source: {source}*\n\n---\n\n"
    
    if not response:
        return ("No information found for the specified entity in the documents you provided.", False)
    
    return (response, True)


def _answer_events_query(data: dict) -> Tuple[str, bool]:
    """Generate response for timeline/events queries."""
    
    events = data.get("events", [])
    
    if not events:
        return ("No dated events were identified in the documents you provided.", True)
    
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
                f"## No Relationships Found in Provided Documents\n\n"
                f"No relationships involving {', '.join(potential_names)} were found in the documents you provided.\n\n"
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


def _answer_motive_query(question: str, data: dict) -> Tuple[str, bool]:
    """
    Generate response for motive/benefit queries using LLM analysis.
    
    Identifies the victim, gathers all connected people and their relationships,
    then uses LLM to analyze potential benefits (financial, practical, emotional).
    """
    import json
    
    # First, identify the victim
    victim = identify_victim(data)
    
    if not victim:
        return (
            "## Unable to Identify Victim\n\n"
            "Could not identify the victim/deceased person in the uploaded documents. "
            "This analysis requires knowing who the victim is to identify who might benefit from the crime.\n\n"
            "Please ensure documents containing victim information have been uploaded.",
            False
        )
    
    victim_name = victim.get("name", "Unknown")
    victim_desc = victim.get("description", "")
    
    # Get all people connected to the victim
    connected_people = get_people_connected_to_victim(data, victim_name)
    
    # Also check for pre-extracted benefit indicators
    benefit_indicators = data.get("benefit_indicators", [])
    
    # Build context for LLM analysis
    relationships = data.get("relationships", [])
    claims = data.get("claims", [])
    events = data.get("events", [])
    
    # Build the raw data response that will be sent to synthesis
    response = f"## Benefit Analysis\n\n"
    response += f"**Victim:** {victim_name}\n"
    if victim_desc:
        response += f"**Background:** {victim_desc}\n"
    response += "\n---\n\n"
    
    # Include pre-extracted benefit indicators if available
    if benefit_indicators:
        response += "### Pre-Extracted Benefit Indicators\n\n"
        for indicator in benefit_indicators:
            person = indicator.get("person", "Unknown")
            benefit_type = indicator.get("type", "unknown")
            subtype = indicator.get("subtype", "")
            description = indicator.get("description", "")
            quote = indicator.get("quote", "")
            source = indicator.get("source", "Unknown")
            
            response += f"**{person}** ({benefit_type}"
            if subtype:
                response += f" - {subtype.replace('_', ' ')}"
            response += ")\n\n"
            if description:
                response += f"{description}\n\n"
            if quote:
                response += f"> \"{quote[:300]}{'...' if len(quote) > 300 else ''}\"\n\n"
            response += f"*Source: {source}*\n\n---\n\n"
    
    # Include connected people for analysis
    if connected_people:
        response += "### People Connected to Victim\n\n"
        response += "_The following individuals have documented relationships with the victim:_\n\n"
        
        for person_data in connected_people:
            name = person_data.get("name", "Unknown")
            entity_info = person_data.get("entity_info", {})
            person_relationships = person_data.get("relationships", [])
            person_claims = person_data.get("claims", [])
            
            response += f"#### {name}\n\n"
            
            if entity_info:
                desc = entity_info.get("description", "")
                if desc:
                    response += f"**Description:** {desc}\n\n"
            
            # Document their relationships
            if person_relationships:
                response += "**Documented Relationships:**\n\n"
                for rel in person_relationships:
                    rel_type = rel.get("type", "unknown").replace("_", " ")
                    rel_desc = rel.get("description", "")
                    rel_quote = rel.get("quote", "")
                    rel_source = rel.get("source", "Unknown")
                    
                    response += f"- **{rel_type}**: {rel_desc}\n"
                    if rel_quote:
                        response += f"  > \"{rel_quote[:200]}{'...' if len(rel_quote) > 200 else ''}\"\n"
                    response += f"  *Source: {rel_source}*\n\n"
            
            # Document relevant claims
            if person_claims:
                response += "**Relevant Statements:**\n\n"
                for claim in person_claims[:5]:  # Limit to 5 most relevant
                    claim_text = claim.get("claim", "")
                    claim_quote = claim.get("quote", "")
                    claim_source = claim.get("source", "Unknown")
                    
                    response += f"- {claim_text}\n"
                    if claim_quote:
                        quote_text = claim_quote if isinstance(claim_quote, str) else (claim_quote[0] if claim_quote else "")
                        if quote_text:
                            response += f"  > \"{str(quote_text)[:200]}{'...' if len(str(quote_text)) > 200 else ''}\"\n"
                    response += f"  *Source: {claim_source}*\n\n"
            
            response += "---\n\n"
    
    # Look for specific benefit-related evidence in claims
    benefit_keywords = {
        "financial": ["money", "insurance", "beneficiary", "inherit", "debt", "owed", "pay", "financial", "estate", "will", "property", "asset"],
        "practical": ["secret", "knew about", "blackmail", "witness", "threat", "obstacle", "control", "power", "business"],
        "emotional": ["jealous", "jealousy", "angry", "hate", "revenge", "affair", "cheating", "ex-", "custody", "divorce", "fight", "argument", "conflict"]
    }
    
    benefit_evidence = {"financial": [], "practical": [], "emotional": []}
    
    for claim in claims:
        claim_text = claim.get("claim", "").lower()
        quote = claim.get("quote", "")
        if isinstance(quote, list):
            quote = quote[0] if quote else ""
        quote_lower = str(quote).lower()
        
        for benefit_type, keywords in benefit_keywords.items():
            if any(kw in claim_text or kw in quote_lower for kw in keywords):
                benefit_evidence[benefit_type].append(claim)
    
    # Add benefit-related evidence sections
    if any(benefit_evidence.values()):
        response += "### Evidence Relevant to Potential Benefits\n\n"
        
        if benefit_evidence["financial"]:
            response += "#### Financial Evidence\n\n"
            for claim in benefit_evidence["financial"][:5]:
                subject = claim.get("subject", "Unknown")
                claim_text = claim.get("claim", "")
                quote = claim.get("quote", "")
                source = claim.get("source", "Unknown")
                
                response += f"**{subject}**: {claim_text}\n"
                if quote:
                    quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
                    if quote_text:
                        response += f"> \"{str(quote_text)[:200]}...\"\n"
                response += f"*Source: {source}*\n\n"
        
        if benefit_evidence["practical"]:
            response += "#### Practical/Strategic Evidence\n\n"
            for claim in benefit_evidence["practical"][:5]:
                subject = claim.get("subject", "Unknown")
                claim_text = claim.get("claim", "")
                quote = claim.get("quote", "")
                source = claim.get("source", "Unknown")
                
                response += f"**{subject}**: {claim_text}\n"
                if quote:
                    quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
                    if quote_text:
                        response += f"> \"{str(quote_text)[:200]}...\"\n"
                response += f"*Source: {source}*\n\n"
        
        if benefit_evidence["emotional"]:
            response += "#### Emotional/Relational Evidence\n\n"
            for claim in benefit_evidence["emotional"][:5]:
                subject = claim.get("subject", "Unknown")
                claim_text = claim.get("claim", "")
                quote = claim.get("quote", "")
                source = claim.get("source", "Unknown")
                
                response += f"**{subject}**: {claim_text}\n"
                if quote:
                    quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
                    if quote_text:
                        response += f"> \"{str(quote_text)[:200]}...\"\n"
                response += f"*Source: {source}*\n\n"
    
    # If we found very little, note the gaps
    if not connected_people and not benefit_indicators and not any(benefit_evidence.values()):
        response = "## Limited Benefit-Related Information\n\n"
        response += f"**Victim identified:** {victim_name}\n\n"
        response += "The documents contain limited information about potential benefits:\n\n"
        response += "- No documented financial relationships (insurance, inheritance, debts)\n"
        response += "- No documented conflicts that might suggest practical benefit\n"
        response += "- No documented emotional conflicts (jealousy, revenge)\n\n"
        response += "**Recommendation:** Consider uploading:\n"
        response += "- Financial records\n"
        response += "- Insurance documentation\n"
        response += "- Witness statements about relationships and conflicts\n"
    
    # Add the mandatory disclaimer
    response += "\n\n" + MOTIVE_DISCLAIMER
    
    return (response, True)


def _build_evidence_context(person_name: str, data: dict) -> dict:
    """
    Build comprehensive evidence context for a specific person.
    
    Gathers all evidence related to this person for follow-up question generation:
    - Their own claims and statements
    - Physical/forensic evidence from the case
    - Other people's statements that mention them
    - Timeline of events they were involved in
    - Conflicts involving their statements
    - Key facts from the case
    
    Args:
        person_name: Name of the target person
        data: Extracted data dictionary
        
    Returns:
        Dictionary with categorized evidence context
    """
    from collections import defaultdict
    
    person_normalized = _normalize_for_matching(person_name)
    person_words = set(person_normalized.split())
    
    context = {
        "target_person": person_name,
        "subject_claims": [],
        "subject_quotes": [],
        "physical_evidence": [],
        "third_party_mentions": [],
        "subject_timeline": [],
        "overlapping_timelines": [],
        "conflicts": [],
        "case_facts": [],
        "relationships": [],
        "documents_analyzed": set()
    }
    
    claims = data.get("claims", [])
    events = data.get("events", [])
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    conflicts = data.get("conflicts", [])
    key_facts = data.get("key_facts", [])
    
    # Helper to check if text matches person
    def matches_person(text: str) -> bool:
        text_normalized = _normalize_for_matching(text)
        text_words = set(text_normalized.split())
        return (person_normalized in text_normalized or 
                text_normalized in person_normalized or
                bool(person_words & text_words))
    
    # 1. Gather the person's own claims and quotes
    for claim in claims:
        subject = claim.get("subject", "")
        if matches_person(subject):
            context["subject_claims"].append(claim)
            
            # Extract quotes
            quote = claim.get("quote", "")
            if quote:
                quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
                if quote_text:
                    context["subject_quotes"].append({
                        "quote": quote_text,
                        "claim": claim.get("claim", ""),
                        "source": claim.get("source", "Unknown")
                    })
            
            # Track documents
            source = claim.get("source", "")
            if source:
                context["documents_analyzed"].add(source)
    
    # 2. Find claims where OTHER people mention this person (third party)
    for claim in claims:
        subject = claim.get("subject", "")
        claim_text = claim.get("claim", "").lower()
        quote = claim.get("quote", "")
        quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
        
        # Skip if this is the person's own claim
        if matches_person(subject):
            continue
        
        # Check if person is mentioned in the claim or quote
        if (matches_person(claim_text) or 
            (quote_text and matches_person(quote_text))):
            context["third_party_mentions"].append(claim)
    
    # 3. Gather events involving the person (their timeline)
    for event in events:
        people_involved = event.get("people_involved", [])
        description = event.get("description", "")
        
        person_involved = any(matches_person(p) for p in people_involved)
        mentioned_in_desc = matches_person(description)
        
        if person_involved or mentioned_in_desc:
            context["subject_timeline"].append(event)
            source = event.get("source", "")
            if source:
                context["documents_analyzed"].add(source)
    
    # 4. Find overlapping events (same time period, different people)
    # Get time references from subject's timeline
    subject_times = set()
    for event in context["subject_timeline"]:
        date = event.get("date", "")
        if date and date.lower() != "unknown":
            subject_times.add(date.lower())
    
    # Find other events at similar times
    for event in events:
        if event in context["subject_timeline"]:
            continue
        date = event.get("date", "")
        if date and date.lower() in subject_times:
            context["overlapping_timelines"].append(event)
    
    # 5. Gather conflicts involving the person
    for conflict in conflicts:
        conflict_claims = conflict.get("claims", [])
        subject = conflict.get("subject", "")
        
        if matches_person(subject):
            context["conflicts"].append(conflict)
            continue
        
        # Check if person is mentioned in any of the conflict claims
        for claim in conflict_claims:
            claim_subject = claim.get("subject", "")
            if matches_person(claim_subject):
                context["conflicts"].append(conflict)
                break
    
    # 6. Gather relationships involving the person
    for rel in relationships:
        p1 = rel.get("person1", "")
        p2 = rel.get("person2", "")
        if matches_person(p1) or matches_person(p2):
            context["relationships"].append(rel)
    
    # 7. Gather physical evidence and key facts
    # Look for forensic/physical evidence in entities
    for entity in entities:
        entity_type = entity.get("type", "").lower()
        description = entity.get("description", "").lower()
        
        # Check for evidence-related entities
        evidence_keywords = ["evidence", "fingerprint", "dna", "weapon", "knife", 
                           "blood", "forensic", "crime scene", "autopsy"]
        if any(kw in entity_type or kw in description for kw in evidence_keywords):
            context["physical_evidence"].append(entity)
    
    # Key facts
    for fact in key_facts:
        if isinstance(fact, dict):
            fact_text = fact.get("fact", "")
            context["case_facts"].append(fact)
        else:
            context["case_facts"].append({"fact": str(fact), "source": "Unknown"})
    
    # Convert set to list for JSON serialization
    context["documents_analyzed"] = list(context["documents_analyzed"])
    
    return context


def _format_evidence_context_for_prompt(context: dict) -> str:
    """
    Format the evidence context into a structured string for the LLM prompt.
    """
    import json
    
    output = []
    person_name = context.get("target_person", "Unknown")
    
    output.append(f"# Evidence Context for {person_name}\n")
    output.append(f"**Documents analyzed:** {', '.join(context.get('documents_analyzed', [])) or 'None'}\n")
    
    # Subject's own claims
    claims = context.get("subject_claims", [])
    if claims:
        output.append(f"\n## {person_name}'s Own Statements ({len(claims)} claims)\n")
        for i, claim in enumerate(claims, 1):
            output.append(f"\n### Claim {i}")
            output.append(f"**Statement:** {claim.get('claim', 'N/A')}")
            quote = claim.get("quote", "")
            if quote:
                quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
                if quote_text:
                    output.append(f"> \"{quote_text[:500]}{'...' if len(str(quote_text)) > 500 else ''}\"")
            output.append(f"*Source: {claim.get('source', 'Unknown')}*")
    
    # Subject's timeline
    timeline = context.get("subject_timeline", [])
    if timeline:
        output.append(f"\n## {person_name}'s Timeline ({len(timeline)} events)\n")
        for event in timeline:
            date = event.get("date", "Unknown date")
            desc = event.get("description", "No description")
            source = event.get("source", "Unknown")
            output.append(f"- **{date}:** {desc} *(Source: {source})*")
    
    # Third party mentions
    third_party = context.get("third_party_mentions", [])
    if third_party:
        output.append(f"\n## What Others Say About {person_name} ({len(third_party)} mentions)\n")
        for claim in third_party:
            subject = claim.get("subject", "Unknown")
            claim_text = claim.get("claim", "N/A")
            source = claim.get("source", "Unknown")
            quote = claim.get("quote", "")
            output.append(f"**{subject}** stated: {claim_text}")
            if quote:
                quote_text = quote if isinstance(quote, str) else (quote[0] if quote else "")
                if quote_text:
                    output.append(f"> \"{quote_text[:300]}{'...' if len(str(quote_text)) > 300 else ''}\"")
            output.append(f"*Source: {source}*\n")
    
    # Overlapping timelines (other witnesses at same times)
    overlapping = context.get("overlapping_timelines", [])
    if overlapping:
        output.append(f"\n## Other Events at Same Times ({len(overlapping)} events)\n")
        for event in overlapping:
            date = event.get("date", "Unknown")
            desc = event.get("description", "No description")
            people = event.get("people_involved", [])
            source = event.get("source", "Unknown")
            output.append(f"- **{date}:** {desc}")
            if people:
                output.append(f"  People involved: {', '.join(people)}")
            output.append(f"  *Source: {source}*")
    
    # Physical evidence
    evidence = context.get("physical_evidence", [])
    if evidence:
        output.append(f"\n## Physical/Forensic Evidence ({len(evidence)} items)\n")
        for item in evidence:
            name = item.get("name", "Unknown")
            desc = item.get("description", "No description")
            source = item.get("source", "Unknown")
            if isinstance(source, list):
                source = ", ".join(source)
            output.append(f"- **{name}:** {desc} *(Source: {source})*")
    
    # Conflicts
    conflicts = context.get("conflicts", [])
    if conflicts:
        output.append(f"\n## Identified Conflicts/Inconsistencies ({len(conflicts)} conflicts)\n")
        for conflict in conflicts:
            conflict_type = conflict.get("type", "unknown")
            description = conflict.get("description", "No description")
            output.append(f"### {conflict_type.replace('_', ' ').title()}")
            output.append(f"{description}")
            
            conflict_claims = conflict.get("claims", [])
            for claim in conflict_claims[:3]:  # Limit to 3 per conflict
                subject = claim.get("subject", "Unknown")
                claim_text = claim.get("claim", "N/A")
                source = claim.get("source", "Unknown")
                output.append(f"- **{subject}:** {claim_text} *(Source: {source})*")
    
    # Relationships
    relationships = context.get("relationships", [])
    if relationships:
        output.append(f"\n## {person_name}'s Relationships ({len(relationships)} documented)\n")
        for rel in relationships:
            p1 = rel.get("person1", "Unknown")
            p2 = rel.get("person2", "Unknown")
            rel_type = rel.get("type", "unknown").replace("_", " ")
            desc = rel.get("description", "")
            source = rel.get("source", "Unknown")
            output.append(f"- **{p1}** ↔ **{p2}** ({rel_type}): {desc} *(Source: {source})*")
    
    # Key case facts
    facts = context.get("case_facts", [])
    if facts:
        output.append(f"\n## Key Case Facts ({len(facts)} facts)\n")
        for fact in facts[:15]:  # Limit to 15
            if isinstance(fact, dict):
                output.append(f"- {fact.get('fact', str(fact))} *(Source: {fact.get('source', 'Unknown')})*")
            else:
                output.append(f"- {fact}")
    
    return "\n".join(output)


def _answer_follow_up_query(
    question: str, 
    data: dict,
    client: OpenAI = None,
    model: str = None
) -> Tuple[str, bool]:
    """
    Generate evidence-driven follow-up questions for a specific person.
    
    Args:
        question: The user's question
        data: Extracted data dictionary
        client: OpenAI client (required for LLM generation)
        model: Model to use
        
    Returns:
        Tuple of (response_text, success)
    """
    from pathlib import Path
    
    entities = data.get("entities", [])
    
    # Extract target person from the question
    target_entity = extract_target_person_for_follow_up(question, entities)
    
    if not target_entity:
        # Try to find any person mentioned
        potential_names = _extract_potential_names(question)
        if potential_names:
            return (
                f"## Could Not Find Person in Provided Documents\n\n"
                f"I couldn't find information about **{', '.join(potential_names)}** in the documents you provided.\n\n"
                f"**People in the documents you provided:**\n" +
                "\n".join(f"- {e.get('name')}" for e in entities if e.get('type', '').lower() == 'person')[:10] +
                "\n\n*Try asking about one of these people instead.*",
                False
            )
        return (
            "## No Person Specified\n\n"
            "Please specify who you want follow-up questions for.\n\n"
            "**Example:** \"Generate follow-up questions for Dennis Rosa Roman\"\n\n"
            "**People in the documents:**\n" +
            "\n".join(f"- {e.get('name')}" for e in entities if e.get('type', '').lower() == 'person')[:10],
            False
        )
    
    person_name = target_entity.get("name", "Unknown")
    
    # Build comprehensive evidence context
    evidence_context = _build_evidence_context(person_name, data)
    
    # Check if we have enough data
    total_evidence = (
        len(evidence_context.get("subject_claims", [])) +
        len(evidence_context.get("subject_timeline", [])) +
        len(evidence_context.get("third_party_mentions", []))
    )
    
    if total_evidence == 0:
        return (
            f"## Insufficient Evidence for {person_name}\n\n"
            f"I don't have enough documented statements or evidence about **{person_name}** "
            f"to generate meaningful follow-up questions.\n\n"
            f"**What I found:**\n"
            f"- Claims by {person_name}: {len(evidence_context.get('subject_claims', []))}\n"
            f"- Events involving {person_name}: {len(evidence_context.get('subject_timeline', []))}\n"
            f"- Mentions by others: {len(evidence_context.get('third_party_mentions', []))}\n\n"
            f"*Try uploading more documents containing statements from or about {person_name}.*",
            False
        )
    
    # If no client provided, return raw context (for testing)
    if not client or not model:
        formatted_context = _format_evidence_context_for_prompt(evidence_context)
        return (
            f"## Evidence Context for {person_name}\n\n"
            f"*(LLM synthesis not available - showing raw evidence)*\n\n"
            f"{formatted_context}",
            True
        )
    
    # Load the follow-up generator prompt
    prompt_path = Path("prompts/follow_up_generator.md")
    if not prompt_path.exists():
        return ("Follow-up generator prompt not found.", False)
    
    follow_up_prompt = prompt_path.read_text()
    
    # Replace placeholder with target person
    follow_up_prompt = follow_up_prompt.replace("{target_person}", person_name)
    
    # Format evidence context
    formatted_context = _format_evidence_context_for_prompt(evidence_context)
    
    # Build the full prompt
    user_prompt = f"""Generate follow-up interview questions for **{person_name}** based on the following evidence.

{formatted_context}

---

Based on this evidence, generate specific, evidence-grounded follow-up questions that:
1. Probe inconsistencies in their statements
2. Seek corroboration for unverified claims
3. Fill timeline gaps
4. Compare their account to other witnesses
5. Connect their statements to physical evidence

Format the output as specified in your instructions."""

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": follow_up_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return (resp.output_text.strip(), True)
    except Exception as e:
        print(f"Follow-up question generation error: {e}")
        return (f"Error generating follow-up questions: {e}", False)


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

