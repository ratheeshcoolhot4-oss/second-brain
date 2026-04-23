# agents/archivist.py
#
# ============================================================
# THE ARCHIVIST — TWO-CALL EXTRACTION ARCHITECTURE
# ============================================================
#
# BEFORE (one call, one giant schema):
#   extract(text) → one API call → ExtractedEvent
#   Problem: token explosion, LLM verbose with nested objects
#
# AFTER (two calls, two focused schemas):
#   _extract_core(text)   → Call 1 → CoreEvent    (~400 tokens)
#   _extract_people(text) → Call 2 → PeopleExtraction (~300 tokens)
#   _merge(core, people)  → ExtractedEvent
#
# INTERNAL — WHY TWO CALLS BEATS ONE BIG CALL:
# ---------------------------------------------
# Each LLM call has its own context window and token budget.
# Two small calls never interfere with each other.
#
# Think of it like a database transaction:
#   Bad:  one giant stored procedure doing everything
#   Good: two focused queries, each doing one thing well
#
# The total token cost is LOWER with two calls because:
#   One big call: LLM preambles, elaborates, reasons across all fields
#   Two small calls: each stays focused on its narrow schema
#   Focused calls → concise output → fewer tokens
#
# CALL FLOW:
# ----------
#   raw_text
#      │
#      ▼
#   Call 1: CoreEvent extraction
#      │    10 flat fields
#      │    Always runs
#      │    ~400 tokens
#      │
#      ├── has_people = False → skip Call 2
#      │                        merge with empty people list
#      │
#      └── has_people = True
#             │
#             ▼
#          Call 2: PeopleExtraction
#                  1 nested list
#                  Only when needed
#                  ~300 tokens
#             │
#             ▼
#          merge core + people → ExtractedEvent
# ============================================================

from openai import OpenAI
from models.schemas import (
    CoreEvent,
    PeopleExtraction,
    ExtractedEvent,
    PersonMention
)
from dotenv import load_dotenv
import os

load_dotenv()


class ArchivistAgent:
    """
    Reads raw journal text and extracts structured events.

    Uses two focused API calls instead of one large call.
    Each call has a small schema → concise output → no token explosion.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model  = "gpt-4o"

        # ── SYSTEM PROMPT: CALL 1 (CORE EXTRACTION) ─────────────
        #
        # INTERNAL — Why the prompt is structured this way:
        #
        # The system prompt sets the LLM's "frame" before it sees
        # any user input. It goes into the transformer context first
        # and influences how every subsequent token is sampled.
        #
        # Key design decisions:
        #
        # 1. "Be extremely concise" at the TOP of rules
        #    The LLM reads rules sequentially. Rules at the top
        #    have higher influence on early token decisions.
        #    Putting conciseness first primes the model before
        #    it starts generating any field values.
        #
        # 2. Character limits in field descriptions
        #    "Max 15 words" in the field description becomes part
        #    of the JSON Schema sent to OpenAI. The model sees
        #    this constraint directly in the grammar specification.
        #
        # 3. Confidence inference rules with examples
        #    LLMs respond better to examples than abstract rules.
        #    "1-2: 'against my gut', 'pressured into'" is more
        #    effective than "1-2: very low confidence scenarios"
        #    because it matches token patterns the model knows.
        self.core_system_prompt = """You are a precise extraction engine for a personal decision intelligence system.

Extract structured data from journal entries. Be extremely concise.

CONFIDENCE INFERENCE — always infer, never skip:
8-10 → "I know", "absolutely", "certain", "ready", "clear"
5-7  → "I think", "should be", "seems", "hope", "worth trying"
3-4  → "not sure", "uneasy", "something feels off", "hesitant"
1-2  → "against my gut", "pressured into", "too tired to think", "couldn't say no"

CRITICAL: Low confidence + fatigue is the most important pattern. Capture it precisely.

OUTPUT RULES:
- action_description: one sentence, max 15 words
- confidence_basis: the exact phrase that determined the score, max 10 words
- external_pressures: 1-3 words each, no sentences
- tags: max 3 tags from: work, health, finance, relationships, habits, family
- has_people: true only if a specific person is mentioned by name or identifier"""

        # ── SYSTEM PROMPT: CALL 2 (PEOPLE EXTRACTION) ───────────
        #
        # Completely separate prompt focused only on people.
        # The LLM has full attention on one task: extract people
        # with their belief and evidence streams.
        #
        # WHY THE BELIEF/EVIDENCE DISTINCTION IS EMPHASIZED:
        # This distinction is subtle and the LLM will conflate them
        # without explicit instruction.
        #
        # "I think X is trustworthy" → stated_belief
        # "X missed the deadline"    → interaction_description
        #
        # Both are sentences about X. The LLM needs to understand:
        # one is your OPINION, one is a BEHAVIORAL FACT.
        # The prompt enforces this with examples, not just definitions.
        self.people_system_prompt = """You extract people mentions from journal entries for a personal intelligence system.

CRITICAL DISTINCTION — never confuse these:
stated_belief      = what the person SAYS or THINKS about someone
                     Example: "I think X is trustworthy" → stated_belief = "X is trustworthy"
                     Example: "X seems reliable" → stated_belief = "X seems reliable"

interaction_description = what the person actually DID (behavior/action)
                          Example: "X missed the deadline" → interaction_description = "missed deadline"
                          Example: "X apologized" → interaction_description = "apologized"

RULES:
- Only populate stated_belief if an opinion/belief about the person is explicitly stated
- Only populate interaction_description if a specific behavior/action by them is described
- Both can be null for the same person if neither is present
- Keep all values under 10 words
- Use the person's name exactly as written in the text"""

    # ================================================================
    # PUBLIC METHOD — This is what the rest of the system calls
    # ================================================================

    def extract(self, raw_text: str) -> ExtractedEvent:
        """
        Full extraction pipeline: two calls → merge → ExtractedEvent

        WHAT HAPPENS:
        1. Call 1: extract core event data (always)
        2. Check has_people flag from Call 1 result
        3. Call 2: extract people data (only if has_people = True)
        4. Merge both results into ExtractedEvent
        5. Attach raw_text (never sent to LLM, always preserved)

        Returns: ExtractedEvent ready for database + vector store
        """

        print(f"\n🔍 Archivist reading: '{raw_text[:60]}...'")

        # ── CALL 1: CORE EXTRACTION ──────────────────────────────
        core = self._extract_core(raw_text)

        # ── CALL 2: PEOPLE (conditional) ─────────────────────────
        people = []
        if core.has_people:
            people_result = self._extract_people(raw_text)
            people = people_result.people
            print(f"   👥 Extracted {len(people)} person(s): "
                  f"{[p.name for p in people]}")
        else:
            print(f"   👥 No people mentioned — skipping Call 2")

        # ── MERGE ────────────────────────────────────────────────
        event = self._merge(core, people, raw_text)

        print(f"   ✓ Complete: [{event.event_type.value}] "
              f"confidence={event.confidence_score}/10 "
              f"emotions={[e.value for e in event.emotional_states]}")

        return event

    # ================================================================
    # PRIVATE METHODS — Internal implementation
    # ================================================================

    def _extract_core(self, raw_text: str) -> CoreEvent:
        """
        Call 1: Extract core event fields from raw text.

        Schema: CoreEvent (10 flat fields, no nesting)
        Expected tokens: 300-500
        Always runs.

        INTERNAL — finish_reason check:
        We check finish_reason BEFORE accessing .parsed
        finish_reason = "stop"   → LLM completed naturally → safe
        finish_reason = "length" → hit token ceiling → JSON incomplete
        finish_reason = "content_filter" → input flagged → no output

        This check prevents silent data corruption.
        If finish_reason != "stop", we raise immediately.
        The database never receives incomplete data.
        """

        print(f"   → Call 1: Core extraction...")

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.core_system_prompt},
                {"role": "user",   "content": f"Extract core event data:\n\n{raw_text}"}
            ],
            response_format=CoreEvent,
            temperature=0,
            max_tokens=1024    # CoreEvent needs ~300-500 tokens. 1024 is safe ceiling.
        )

        self._check_finish_reason(response, call_name="Core extraction")

        core = response.choices[0].message.parsed

        if core is None:
            raise ValueError("Core extraction returned None. Check schema compatibility.")

        print(f"   ✓ Call 1 done: {response.usage.completion_tokens} tokens used "
              f"(budget: 1024)")

        return core

    def _extract_people(self, raw_text: str) -> PeopleExtraction:
        """
        Call 2: Extract people mentions from raw text.

        Schema: PeopleExtraction (list of PersonMention)
        Expected tokens: 200-400
        Only runs when has_people = True from Call 1.

        INTERNAL — Why we pass raw_text again and not just names:
        Call 1 told us people are present but didn't extract them.
        Call 2 needs the full text to extract:
          - which person said/did what
          - the belief vs evidence distinction
          - sentiment for each stream

        Passing just names would lose the context needed to
        separate belief from evidence correctly.
        """

        print(f"   → Call 2: People extraction...")

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.people_system_prompt},
                {"role": "user",   "content": f"Extract all people mentions:\n\n{raw_text}"}
            ],
            response_format=PeopleExtraction,
            temperature=0,
            max_tokens=1024    # PeopleExtraction needs ~200-400 tokens. 1024 is safe.
        )

        self._check_finish_reason(response, call_name="People extraction")

        people = response.choices[0].message.parsed

        if people is None:
            raise ValueError("People extraction returned None.")

        print(f"   ✓ Call 2 done: {response.usage.completion_tokens} tokens used "
              f"(budget: 1024)")

        return people

    def _merge(self,
               core: CoreEvent,
               people: list[PersonMention],
               raw_text: str) -> ExtractedEvent:
        """
        Merge Call 1 + Call 2 results into a single ExtractedEvent.

        INTERNAL — Why not just use CoreEvent directly?
        CoreEvent is the extraction schema — it has has_people (a flag
        used to decide whether to run Call 2) but no people list.
        ExtractedEvent is the domain model — it has people but no has_people.

        The merge step:
          1. Takes all fields from CoreEvent
          2. Adds people list from PeopleExtraction (or empty list)
          3. Adds raw_text (preserved, never modified)
          4. Produces ExtractedEvent which database/vector store expect

        raw_text is attached HERE, not in the LLM calls.
        Why? We never want the LLM to re-summarize or modify your words.
        raw_text is the source of truth. We preserve it exactly.
        """

        return ExtractedEvent(
            # From Call 1
            event_type          = core.event_type,
            action_description  = core.action_description,
            emotional_states    = core.emotional_states,
            external_pressures  = core.external_pressures,
            confidence_score    = core.confidence_score,
            confidence_basis    = core.confidence_basis,
            decision_pressure   = core.decision_pressure,
            event_date          = core.event_date,
            tags                = core.tags,
            relates_to_decision = core.relates_to_decision,
            decision_context    = core.decision_context,
            decision_alternatives = core.decision_alternatives,
            embedding_summary   = core.embedding_summary,

            # From Call 2 (or empty)
            people              = people,

            # Always preserved exactly
            raw_text            = raw_text,
        )

    def _check_finish_reason(self, response, call_name: str):
        """
        Verify the LLM completed cleanly before trusting the output.

        INTERNAL — The three finish reasons we care about:

        "stop"          → LLM hit a natural end token
                          JSON is complete and valid
                          Safe to parse

        "length"        → LLM hit max_tokens ceiling
                          JSON was cut off mid-generation
                          .parsed will be None or incomplete
                          We must raise — never store partial data

        "content_filter"→ OpenAI's safety system blocked the output
                          Rare for journal entries
                          Raise with clear message

        WHY THIS CHECK MATTERS:
        Without it, if finish_reason = "length":
          response.choices[0].message.parsed → None
          _merge receives None → AttributeError deep in the code
          Stack trace points to merge, not the real problem (token limit)
          Debugging is confusing

        With it:
          Raises immediately with clear message + token count
          You know exactly what happened and why
          Fix: increase max_tokens or simplify the schema
        """

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            used   = response.usage.completion_tokens
            budget = 1024
            raise ValueError(
                f"{call_name} hit token limit. "
                f"Used {used}/{budget} tokens. "
                f"Input may be unusually long. "
                f"Consider splitting into shorter entries."
            )

        if finish_reason == "content_filter":
            raise ValueError(
                f"{call_name} was blocked by content filter. "
                f"Try rephrasing the input."
            )

        if finish_reason != "stop":
            raise ValueError(
                f"{call_name} ended with unexpected finish_reason='{finish_reason}'"
            )


# ============================================================
# STANDALONE TEST
# Run: python agents/archivist.py
# Tests all three scenarios:
#   1. Simple entry, no people
#   2. Entry with people + beliefs + behavior
#   3. Decision with low confidence signals
# ============================================================

if __name__ == "__main__":
    from rich import print as rprint

    agent = ArchivistAgent()

    test_entries = [
        # Scenario 1: No people, emotion + decision
        "Accepted extra work today. Feeling really tired. "
        "Money pressure is real. Said yes even though something felt off.",

        # Scenario 2: People mentioned — belief + evidence streams
        "Had coffee with Ravi today. He promised to connect me with "
        "his network by Friday. I think he's a genuinely good person "
        "and I trust him completely.",

        # Scenario 3: Low confidence, external pressure
        "My manager pushed me to take Project Alpha. "
        "I was exhausted and couldn't think straight. "
        "Agreed even though my gut said no. "
        "Alternatives were: delay by a month, or take only Phase 1.",
    ]

    print("\n" + "="*60)
    print("  ARCHIVIST — TWO-CALL EXTRACTION TEST")
    print("="*60)

    for i, entry in enumerate(test_entries, 1):
        print(f"\n{'─'*60}")
        print(f"TEST {i}: {entry[:70]}...")
        print(f"{'─'*60}")

        result = agent.extract(entry)

        print(f"\n📦 EXTRACTED:")
        rprint(result.model_dump())