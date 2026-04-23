# models/schemas.py
#
# ============================================================
# SCHEMA ARCHITECTURE — WHY WE SPLIT
# ============================================================
#
# BEFORE (one giant schema):
#   ExtractedEvent had 14+ fields including nested PersonMention
#   One API call tried to fill everything at once
#   LLM became verbose → 4096-16384 tokens → crash
#
# AFTER (two focused schemas):
#   CoreEvent       → 10 flat fields, no nesting (~300-400 tokens)
#   PeopleExtraction → only PersonMention list  (~200-300 tokens)
#
# WHY SPLITTING FIXES THE TOKEN PROBLEM:
# ---------------------------------------
# The token explosion happened because of TWO things combined:
#
#   1. NESTED OBJECTS: PersonMention inside ExtractedEvent
#      forces the LLM to context-switch between two schemas
#      simultaneously. It generates more tokens to "reason through"
#      the nesting before filling values.
#
#   2. TOO MANY FIELDS: 14+ fields in one schema makes the LLM
#      treat each field as an opportunity to elaborate.
#      Fewer fields = less elaboration temptation.
#
# With split schemas:
#   Call 1: 10 flat fields → LLM stays concise → ~400 tokens
#   Call 2: 1 list of simple objects → focused → ~300 tokens
#   Total:  ~700 tokens vs 4096-16384 before
#
# INTERNAL — HOW PYDANTIC SCHEMA TRANSLATES TO JSON SCHEMA:
# ----------------------------------------------------------
# When you pass response_format=CoreEvent to OpenAI,
# the SDK calls CoreEvent.model_json_schema() internally.
# This produces a JSON Schema dict like:
# {
#   "type": "object",
#   "properties": {
#     "event_type": {"enum": ["decision", "action", ...]},
#     "confidence_score": {"type": "integer", "minimum": 1, "maximum": 10}
#   },
#   "required": ["event_type", "action_description", ...]
# }
# OpenAI uses this schema for grammar-constrained decoding.
# Smaller schema = simpler grammar = fewer tokens needed.
# ============================================================

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from datetime import datetime


# ============================================================
# ENUMS — Controlled vocabularies
#
# INTERNAL: Why enums instead of plain strings?
#
# If we say: emotional_states: list[str]
# LLM might produce: ["exhausted", "worn out", "fatigued"]
# Three different strings, all meaning "tired"
# Pattern detection sees THREE different emotions
# "tired" pattern never accumulates → patterns missed
#
# If we say: emotional_states: list[EmotionalState]
# LLM MUST pick from the enum values
# Always "tired", never "exhausted" or "worn out"
# Pattern detection sees ONE consistent signal
# Patterns accumulate correctly
#
# Enums enforce a controlled vocabulary at the token level.
# The grammar constraint literally makes other tokens invalid.
# ============================================================

class EventType(str, Enum):
    DECISION    = "decision"     # You made a choice
    ACTION      = "action"       # You did something
    EMOTION     = "emotion"      # You felt something
    THOUGHT     = "thought"      # You reflected on something
    OUTCOME     = "outcome"      # Result of a past decision
    INTERACTION = "interaction"  # Something happened with a person


class EmotionalState(str, Enum):
    TIRED       = "tired"
    STRESSED    = "stressed"
    CALM        = "calm"
    ANXIOUS     = "anxious"
    CONFIDENT   = "confident"
    PRESSURED   = "pressured"
    HAPPY       = "happy"
    FRUSTRATED  = "frustrated"
    NEUTRAL     = "neutral"
    OVERWHELMED = "overwhelmed"


class PersonRelationType(str, Enum):
    WORK     = "work"
    PERSONAL = "personal"
    FAMILY   = "family"
    UNKNOWN  = "unknown"


# ============================================================
# SCHEMA 1 — CoreEvent
#
# Used in: Call 1 (always runs)
# Fields: 10 flat fields, zero nesting
# Expected tokens: 300-500
#
# WHY NO PEOPLE HERE:
# People extraction requires nested objects (PersonMention).
# Nesting is the main driver of token explosion.
# We isolate it to Call 2, which only runs when needed.
#
# has_people is a simple boolean flag — the LLM just says
# true or false. This decides whether Call 2 runs at all.
# If no people mentioned → Call 2 never happens → saves tokens + money.
# ============================================================

class CoreEvent(BaseModel):
    event_type: EventType = Field(
        description="Type: decision, action, emotion, thought, outcome, interaction"
    )

    action_description: str = Field(
        description="One concise sentence. Max 15 words. What happened or was decided."
    )

    emotional_states: list[EmotionalState] = Field(
        default=[],
        description="Emotional states present. Pick from enum only."
    )

    external_pressures: list[str] = Field(
        default=[],
        description="External factors: money, deadline, family, work. 1-3 words each."
    )

    # ── CONFIDENCE (always inferred, never null) ──────────────
    # INTERNAL: We changed from Optional[int] to int with default=5
    # Why? Optional allows null → pattern engine has no signal
    # int with default=5 means: if truly unclear, assume medium confidence
    # But the LLM should always infer from language signals, not default
    confidence_score: int = Field(
        default=5,
        description=(
            "Inferred confidence 1-10. ALWAYS infer from language. Never leave at default.\n"
            "8-10: decisive language — 'I know', 'absolutely', 'certain'\n"
            "5-7:  cautious language — 'I think', 'should be', 'hope'\n"
            "3-4:  doubt language   — 'not sure', 'uneasy', 'hesitant'\n"
            "1-2:  override signals — 'against my gut', 'pressured into', 'too tired to think'"
        )
    )

    confidence_basis: str = Field(
        description="The specific phrase that determined confidence score. Max 10 words."
    )

    # ── DECISION PRESSURE ────────────────────────────────────
    # INTERNAL: Why separate from emotional_states?
    # Pressure is EXTERNAL SOURCE, emotion is INTERNAL STATE.
    # "tired" = how you feel inside
    # "external" = someone else is pushing you
    # These are different causal factors for bad decisions.
    # Pattern: tired + external pressure → worst outcomes
    decision_pressure: Optional[str] = Field(
        default=None,
        description="Source of pressure if present: internal, external, circumstantial, or none"
    )

    has_people: bool = Field(
        description="True if any person is mentioned. Controls whether people extraction runs."
    )

    event_date: Optional[str] = Field(
        default=None,
        description="Date if mentioned in text. ISO format YYYY-MM-DD only."
    )

    tags: list[str] = Field(
        default=[],
        description="1-3 tags: work, health, finance, relationships, habits"
    )

    relates_to_decision: Optional[str] = Field(
        default=None,
        description="If this is an outcome: describe the original decision in max 10 words."
    )

    decision_context: Optional[str] = Field(
        default=None,
        description="Situation leading to this decision. Max 20 words. Null if not a decision."
    )

    decision_alternatives: list[str] = Field(
        default=[],
        description="Other options considered. 1-5 words each. Empty if none mentioned."
    )

    embedding_summary: str = Field(
    description=(
        "A normalized description for similarity matching. "
        "Describe the ACTION PATTERN and EMOTIONAL CONTEXT only. "
        "Strip all proper nouns, project names, people names. "
        "Focus on: what category of action, what emotional state, what pressure. "
        "Example: 'accepted additional work responsibility under financial pressure while fatigued' "
        "Example: 'agreed to commitment despite feeling uncertain and externally pressured' "
        "Example: 'skipped health habit due to time pressure and low energy' "
        "Max 20 words."
    )
)

    # ── VALIDATOR ────────────────────────────────────────────
    # INTERNAL: field_validator runs AFTER the LLM response is
    # deserialized from JSON into Python. It's a post-parse hook.
    # If validation fails here, Pydantic raises ValidationError
    # before the object is returned to your code.
    # The database never sees an invalid confidence score.
    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v):
        if not (1 <= v <= 10):
            raise ValueError(f"Confidence must be 1-10, got {v}")
        return v


# ============================================================
# SCHEMA 2 — PeopleExtraction
#
# Used in: Call 2 (only runs when has_people = True)
# Fields: one list of PersonMention objects
# Expected tokens: 200-400
#
# WHY A SEPARATE SCHEMA FOR PEOPLE:
# PersonMention has 6 fields. If we had 3 people mentioned,
# that's 18 field-fills just for people, on top of CoreEvent.
# Isolating to Call 2 means:
#   - Call 1 stays fast and cheap regardless of people
#   - Call 2 only runs when actually needed
#   - Each call has a focused, small schema
#
# THE BELIEF vs EVIDENCE SPLIT:
# This is the core of your contradiction detection feature.
#
# stated_belief      = what you SAY about the person
#                      "I think Ravi is trustworthy"
#                      This is OPINION. Subject to bias.
#
# interaction_description = what ACTUALLY HAPPENED
#                           "Ravi missed the deadline again"
#                           This is EVIDENCE. Objective record.
#
# Critic Agent compares these two streams over time.
# When belief says "trustworthy" but evidence shows 3 misses
# → contradiction flagged → you are warned before trusting again.
# ============================================================

class PersonMention(BaseModel):
    name: str = Field(
        description="Person's name or identifier as written in the text"
    )

    relation_type: PersonRelationType = Field(
        default=PersonRelationType.UNKNOWN,
        description="Relationship type: work, personal, family, or unknown"
    )

    # BELIEF STREAM — what you say/think about them
    stated_belief: Optional[str] = Field(
        default=None,
        description=(
            "ONLY populate if person explicitly stated or implied as good/bad/trustworthy etc. "
            "Exact belief in max 10 words. Null if no belief expressed."
        )
    )
    belief_sentiment: Optional[str] = Field(
        default=None,
        description="positive, negative, or neutral. Only if stated_belief is populated."
    )

    # EVIDENCE STREAM — what actually happened
    interaction_description: Optional[str] = Field(
        default=None,
        description=(
            "ONLY populate if a specific behavior or action by this person is described. "
            "What they DID in max 10 words. Null if no specific behavior described."
        )
    )
    interaction_sentiment: Optional[str] = Field(
        default=None,
        description="positive, negative, or neutral based on their behavior."
    )


class PeopleExtraction(BaseModel):
    # ── WHY A WRAPPER CLASS? ─────────────────────────────────
    # We could pass list[PersonMention] directly to OpenAI.
    # But OpenAI's structured output requires the top level
    # to be an object (dict), not an array (list).
    # Wrapping in a class with one field satisfies this constraint.
    people: list[PersonMention] = Field(
        description="All people mentioned with their belief and evidence streams."
    )


# ============================================================
# EXTRACTED EVENT — The merged result after both calls
#
# This is what the rest of the system (database, vector store,
# agents) works with. It looks identical to the old ExtractedEvent
# so nothing downstream needs to change.
#
# INTERNAL: Why keep this separate from CoreEvent?
# CoreEvent = what Call 1 returns (no people, no raw_text)
# ExtractedEvent = CoreEvent + people + raw_text merged together
#
# The merge happens in archivist.py after both calls complete.
# This class is the clean output of the full extraction process.
# ============================================================

class ExtractedEvent(BaseModel):
    # From CoreEvent
    event_type:          EventType
    action_description:  str
    emotional_states:    list[EmotionalState]  = []
    external_pressures:  list[str]             = []
    confidence_score:    int                   = 5
    confidence_basis:    str                   = ""
    decision_pressure:   Optional[str]         = None
    event_date:          Optional[str]         = None
    tags:                list[str]             = []
    relates_to_decision: Optional[str]         = None
    decision_context:    Optional[str]         = None
    decision_alternatives: list[str]           = []
    embedding_summary:   Optional[str]         = None

    # Added after merge
    raw_text:            str                   = ""
    people:              list[PersonMention]   = []


# ============================================================
# STORED EVENT — ExtractedEvent + database metadata
#
# ExtractedEvent = what the extraction pipeline produces
# StoredEvent    = ExtractedEvent + id + timestamps
#
# Single responsibility principle:
#   Extraction layer doesn't know about database IDs
#   Database layer doesn't know about LLM prompts
#   Each layer only knows what it needs
# ============================================================

class StoredEvent(ExtractedEvent):
    id:           str      = Field(description="UUID assigned at storage time")
    created_at:   datetime = Field(default_factory=datetime.now)
    embedding_id: Optional[str] = Field(default=None)


# ============================================================
# CONTRADICTION RECORD — Produced by Critic Agent
#
# Stored separately because:
#   1. Derived data (computed from events, not input directly)
#   2. Different query patterns (by person, by severity)
#   3. Accumulates over time independently of new events
# ============================================================

class ContradictionRecord(BaseModel):
    person_name:        str
    belief_statement:   str            # "I think X is trustworthy"
    belief_date:        str            # When you said this
    evidence_summary:   str            # "3 missed commitments, 1 credit-taking"
    evidence_event_ids: list[str]      # Which stored events are the evidence
    severity:           str            # "low", "medium", "high"
    detected_at:        datetime = Field(default_factory=datetime.now)


# ============================================================
# PHASE 2 SCHEMAS — RAG Query Response
# ============================================================
#
# These schemas define STRUCTURED OUTPUT from the reasoning layer.
# Same mechanism as extraction — grammar-constrained via Pydantic.
#
# WHY STRUCTURED RESPONSE AND NOT FREE TEXT:
# -------------------------------------------
# Free text: "You might want to be careful, tiredness affects decisions..."
#   → Vague. Unverifiable. Could be hallucinated.
#   → No confidence score. No cited evidence. Can't be stored.
#
# Structured response:
#   pattern:       "accept work when fatigued + pressured"
#   evidence:      [EvidenceItem(event_id="abc123", ...)]
#   confidence:    0.74
#   warning_level: "high"
#   → Every field verifiable against stored data
#   → Evidence cites real event IDs — hallucination structurally hard
#   → Confidence is a number, not "fairly confident"
#   → Can be stored, compared, tracked over time
# ============================================================
 
class EvidenceItem(BaseModel):
    """
    One cited past event in a RAG response.
 
    INTERNAL: Why force the LLM to fill event_id?
    If we just ask for a description, the LLM can invent
    plausible-sounding history. Forcing a specific event_id
    means it must reference something real that exists in our DB.
    After the response, we can verify: does this ID exist?
    If not → hallucination detected.
    This is the core hallucination prevention mechanism.
    """
    event_id:    str            = Field(description="ID of the stored event being cited")
    event_date:  str            = Field(description="When this event occurred")
    description: str            = Field(description="What happened. Max 15 words.")
    outcome:     Optional[str]  = Field(
        default=None,
        description="What resulted. Max 10 words. Null if outcome not yet known."
    )
    relevance:   str            = Field(
        description="Why this event is relevant to the current query. Max 10 words."
    )
 
 
class QueryResponse(BaseModel):
    """
    Structured output from the RAG reasoning pipeline.
 
    INTERNAL — Two different confidence scores in this system:
 
    1. ExtractedEvent.confidence_score (int 1-10)
       = how sure YOU were when making the decision
       = inferred from your language at the time
       = stored with the event
 
    2. QueryResponse.confidence (float 0.0-1.0)
       = how confident the SYSTEM is in the pattern it found
       = computed from: sample_size + outcome_ratio + similarity_avg
       = returned with each query response
 
    These are completely different things.
    One is about your mental state. One is about statistical reliability.
    """
 
    pattern_detected: Optional[str] = Field(
        default=None,
        description=(
            "Behavioral pattern found across retrieved events. "
            "Max 20 words. Null if no clear pattern."
        )
    )
 
    evidence: list[EvidenceItem] = Field(
        default=[],
        description="Specific past events supporting the pattern. Must cite real event IDs."
    )
 
    recommendation: str = Field(
        description=(
            "Specific, actionable recommendation. Max 20 words. "
            "Not 'be careful' — instead: 'delay 7 days' or 'reduce scope by 30%'."
        )
    )
 
    # ── WARNING LEVEL ─────────────────────────────────────────
    # INTERNAL: Why an enum-like string and not just confidence?
    # Confidence is a number (0.74). Warning level is a category.
    # The UI needs a category to decide color/icon:
    #   high   → red warning icon
    #   medium → orange
    #   low    → yellow
    #   none   → green checkmark
    # A float alone can't drive that cleanly.
    warning_level: str = Field(
        description="none, low, medium, or high. Based on outcome ratio in evidence."
    )
 
    confidence: float = Field(
        description=(
            "System confidence in this response. 0.0-1.0. "
            "0.8-1.0: strong pattern, 4+ events, consistent outcomes. "
            "0.5-0.7: moderate, 2-3 events. "
            "0.2-0.4: weak, few events. "
            "0.0-0.2: insufficient data."
        )
    )
 
    reasoning: str = Field(
        description=(
            "Why you reached this recommendation. "
            "Reference specific evidence. Max 40 words."
        )
    )
 
    insufficient_data: bool = Field(
        default=False,
        description="True if fewer than 2 similar events found. Pattern cannot be trusted."
    )
 
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got {v}")
        return round(v, 2)
 
    @field_validator('warning_level')
    @classmethod
    def validate_warning(cls, v):
        valid = {"none", "low", "medium", "high"}
        if v not in valid:
            raise ValueError(f"warning_level must be one of {valid}, got '{v}'")
        return v
 
 
# ============================================================
# LANGGRAPH STATE
# ============================================================
#
# INTERNAL: What is GraphState and why TypedDict not Pydantic?
#
# LangGraph passes state between nodes as a plain Python dict.
# Its internal execution engine expects dict, not Pydantic objects.
#
# TypedDict = a dict with type hints. Gives you:
#   → IDE autocomplete on state keys
#   → Type checking (mypy, pyright)
#   → No runtime validation overhead (unlike Pydantic)
#   → Still a plain dict under the hood (LangGraph compatible)
#
# POPULATION TIMELINE — who writes what:
#   initial call  → query, input_type (empty string)
#   router_node   → input_type (confirmed)
#   retrieve_node → retrieved_events, similarity_scores
#   analyze_node  → pattern, outcome_ratio, sample_size
#   advise_node   → response (QueryResponse object)
#   respond_node  → final_output (formatted string for display)
#
# WHY ONE SHARED STATE (not separate objects per node)?
# Any node can read anything any previous node wrote.
# advise_node reads retrieved_events from retrieve_node
# without being explicitly passed that data.
# The graph structure manages data flow — not function arguments.
# ============================================================
 
from typing import TypedDict
 
 
class GraphState(TypedDict):
    # ── INPUT ─────────────────────────────────────────────────
    query:       str    # Your raw question or statement
    input_type:  str    # "question", "outcome", "log" — set by router_node
 
    # ── RETRIEVAL ─────────────────────────────────────────────
    retrieved_events:  list[dict]    # Full event dicts from SQLite
    similarity_scores: list[float]   # Parallel list — one score per event
 
    # ── ANALYSIS ──────────────────────────────────────────────
    pattern:       Optional[str]   # Dominant behavioral pattern detected
    outcome_ratio: float           # negative_outcomes / total_known_outcomes
    sample_size:   int             # How many similar events found
 
    # ── RESPONSE ──────────────────────────────────────────────
    response:     Optional[QueryResponse]   # Structured LLM output
    final_output: str                       # Human-readable formatted string
 
    # ── ERROR ─────────────────────────────────────────────────
    error: Optional[str]    # If any node fails, reason stored here
                            # Lets respond_node show a clean error message
                            # instead of a Python traceback
 
 
# ============================================================
# PHASE 2 SCHEMAS — RAG Query Response
# ============================================================
#
# These schemas define the STRUCTURED OUTPUT from the reasoning
# layer. Just like extraction, we use Pydantic + structured output
# so the LLM cannot return vague or unverifiable answers.
#
# WHY STRUCTURED RESPONSE (not free text)?
# -----------------------------------------
# Free text response: "You might want to be careful here because
#   sometimes when people are tired they make poor decisions..."
#   → Vague. Unverifiable. Could be hallucinated.
#   → Can't be stored. Can't be compared later.
#   → No confidence score. No cited evidence.
#
# Structured response:
#   pattern:          "accept work when fatigued + pressured"
#   evidence_used:    ["event_id_1", "event_id_2", "event_id_3"]
#   recommendation:   "delay 7 days or reduce scope"
#   confidence:       0.74
#   warning_level:    "high"
#   → Every field verifiable against stored data
#   → Can be stored as a record
#   → Confidence is computed, not guessed
#   → Evidence is cited, not invented
# ============================================================
 
class EvidenceItem(BaseModel):
    """
    One piece of evidence used in a RAG response.
 
    INTERNAL: Why store evidence separately from the response?
    The LLM must cite SPECIFIC past events, not summarize vaguely.
    Forcing it to fill event_id means it must reference real stored
    data, not invent plausible-sounding history.
    This is the core hallucination prevention mechanism.
    """
    event_id:    str = Field(description="ID of the stored event being cited")
    event_date:  str = Field(description="When this event occurred")
    description: str = Field(description="What happened. Max 15 words.")
    outcome:     Optional[str] = Field(
        default=None,
        description="What resulted from this. Max 10 words. Null if unknown."
    )
    relevance:   str = Field(
        description="Why this event is relevant to the current query. Max 10 words."
    )
 
 
class QueryResponse(BaseModel):
    """
    Structured output from the RAG reasoning pipeline.
 
    Every field is verifiable. Nothing vague.
    The LLM fills this schema — grammar-constrained like extraction.
 
    INTERNAL: confidence_score here is DIFFERENT from event confidence_score.
    Event confidence_score = how sure YOU were when making the decision (1-10 int)
    QueryResponse confidence = how confident the SYSTEM is in its pattern (0.0-1.0 float)
 
    System confidence is computed from:
      sample_size:    more similar events → higher confidence
      outcome_ratio:  more negative outcomes → clearer pattern
      similarity_avg: how similar the retrieved events really are
    """
 
    # What pattern was found
    pattern_detected: Optional[str] = Field(
        default=None,
        description=(
            "The behavioral pattern found across retrieved events. "
            "Max 20 words. Null if no clear pattern found."
        )
    )
 
    # The evidence (cited past events)
    evidence: list[EvidenceItem] = Field(
        default=[],
        description="Specific past events that support the pattern. Must cite real event IDs."
    )
 
    # The recommendation
    recommendation: str = Field(
        description=(
            "Specific, actionable recommendation based on the pattern. "
            "Max 20 words. If no pattern: general guidance."
        )
    )
 
    # Warning level — drives UI display
    warning_level: str = Field(
        description="none, low, medium, or high. Based on outcome ratio in evidence."
    )
 
    # System confidence in this response
    confidence: float = Field(
        description=(
            "Confidence in this response 0.0-1.0. "
            "Based on: number of similar events found, outcome consistency, similarity scores. "
            "Low if fewer than 3 similar events found."
        )
    )
 
    # Reasoning transparency
    reasoning: str = Field(
        description=(
            "Why you reached this recommendation. "
            "Reference specific evidence. Max 40 words."
        )
    )
 
    # Was enough history found to be meaningful?
    insufficient_data: bool = Field(
        default=False,
        description="True if fewer than 2 similar events found. Pattern cannot be trusted."
    )
 
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got {v}")
        return round(v, 2)
 
    @field_validator('warning_level')
    @classmethod
    def validate_warning(cls, v):
        valid = {"none", "low", "medium", "high"}
        if v not in valid:
            raise ValueError(f"warning_level must be one of {valid}, got '{v}'")
        return v
 
 
# ============================================================
# LANGGRAPH STATE — The shared state flowing through all nodes
#
# INTERNAL: What is LangGraph state?
#
# LangGraph is a state machine. Every node in the graph:
#   1. Reads from state
#   2. Does its job
#   3. Writes updates back to state
#   4. Passes updated state to next node
#
# State is a TypedDict — a Python dict with typed keys.
# TypedDict gives you type hints without Pydantic overhead.
# Why not Pydantic here? LangGraph needs plain dicts internally
# for its graph execution engine. TypedDict bridges type safety
# and dict compatibility.
#
# WHY ONE SHARED STATE (not per-node objects)?
# If each node had its own object, passing data between nodes
# would require explicit handoffs. With shared state, any node
# can read anything any previous node wrote.
# The Advise node can see what the Retrieve node found without
# being explicitly passed that data.
# ============================================================
 
from typing import TypedDict, Annotated
import operator
 
 
class GraphState(TypedDict):
    """
    Shared state flowing through the LangGraph reasoning graph.
 
    POPULATION TIMELINE:
      At graph start:     query, input_type
      After Router:       input_type confirmed
      After Retrieve:     retrieved_events, similarity_scores
      After Analyze:      pattern, outcome_ratio, sample_size
      After Advise:       response (QueryResponse)
      After Respond:      final_output (formatted string)
    """
 
    # ── INPUT ────────────────────────────────────────────────
    query:        str           # Your raw question or statement
    input_type:   str           # "question", "outcome", "log"
                                # Set by Router node
 
    # ── RETRIEVAL RESULTS ────────────────────────────────────
    retrieved_events:  list[dict]   # Full event dicts from SQLite
    similarity_scores: list[float]  # Cosine similarity per event
                                    # Parallel list to retrieved_events
 
    # ── ANALYSIS RESULTS ─────────────────────────────────────
    pattern:       Optional[str]   # Detected behavioral pattern
    outcome_ratio: float           # negative_outcomes / total_outcomes
    sample_size:   int             # How many similar events found
 
    # ── RESPONSE ─────────────────────────────────────────────
    response:      Optional[QueryResponse]  # Structured LLM response
    final_output:  str                      # Formatted for display
 
    # ── ERROR HANDLING ───────────────────────────────────────
    error:         Optional[str]   # If any node fails, reason stored here
    pending_decision_id: Optional[str] 
 