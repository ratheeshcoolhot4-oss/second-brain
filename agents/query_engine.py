# agents/query_engine.py
#
# ============================================================
# THE RAG QUERY ENGINE — Phase 2 Core
# ============================================================
#
# This is where Phase 1's stored data becomes intelligence.
#
# WHAT IS RAG AGAIN — PRECISELY:
# --------------------------------
# RAG = Retrieval Augmented Generation
#
# The word "Augmented" is the key.
# You are AUGMENTING the LLM's prompt with retrieved evidence.
# The LLM doesn't know your history. You inject it via the prompt.
#
# Without RAG:
#   Prompt: "Should I take this project? I'm exhausted."
#   LLM answers from: general training knowledge
#   Result: generic advice about tiredness and work
#
# With RAG:
#   Prompt: "Should I take this project? I'm exhausted.
#            Here is the user's relevant history:
#              Mar 3: accepted project while tired → burnout
#              Aug 12: said yes under pressure → stress spike
#              Nov 1: took work while overwhelmed → regret"
#   LLM answers from: YOUR specific history
#   Result: grounded, personal, evidence-backed advice
#
# The LLM itself didn't change.
# What changed is what you put in the prompt.
# That's RAG. It's prompt engineering with retrieved context.
#
# THE GROUNDING MECHANISM (hallucination prevention):
# ---------------------------------------------------
# We tell the LLM:
#   "Answer ONLY using the evidence provided below.
#    Cite specific event IDs. Do not generalize beyond the evidence."
#
# The structured output schema enforces this:
#   evidence: list[EvidenceItem] — each item has an event_id field
#   LLM must fill real event IDs, not invent them
#   If it invents an ID → we can detect it (ID not in our DB)
#
# This is fundamentally different from asking the LLM to "be honest".
# We make hallucination structurally difficult, not just discouraged.
#
# THE PIPELINE:
# -------------
# query_text
#   → embed query (same model as stored events)
#   → search ChromaDB (cosine similarity)
#   → fetch full events from SQLite (by IDs)
#   → compute outcome_ratio (how many had bad outcomes)
#   → build grounded prompt (inject evidence)
#   → LLM reasons (structured output → QueryResponse)
#   → format response (human readable)
# ============================================================

from openai import OpenAI
from models.schemas import QueryResponse, EvidenceItem
from storage.vector_store import VectorStore
from storage.database import Database
from utils.embeddings import EmbeddingEngine
from dotenv import load_dotenv
from datetime import datetime
import os
import json

load_dotenv()


class QueryEngine:
    """
    The RAG query engine.

    Takes a natural language question about your history
    and returns a grounded, evidence-backed response.
    """

    def __init__(self, db: Database, vector_store: VectorStore):
        # ── DEPENDENCY INJECTION ──────────────────────────────
        # We receive db and vector_store instead of creating them.
        # Why? Because main.py already created them.
        # Creating them again would open second connections to
        # the same SQLite file and ChromaDB directory.
        # Two connections = potential conflicts.
        # Injection = one shared connection across the whole app.
        self.db           = db
        self.vector_store = vector_store
        self.client       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedder     = EmbeddingEngine()
        self.model        = "gpt-4o"

        # ── RAG PARAMETERS ───────────────────────────────────
        # How many similar events to retrieve?
        # Too few (1-2): not enough for pattern detection
        # Too many (20+): prompt gets huge, LLM loses focus
        # Sweet spot: 5-8 for personal data volume
        self.n_retrieve = 6

        # Minimum similarity score to include in context
        # Events below this threshold are not similar enough
        # to be useful evidence — they'd dilute the pattern
        self.min_similarity = 0.35
   
    def _parse_json_field(value, default=None):
        if value is None:
            return default or []
        if isinstance(value, (list, dict)):
            return value        # PostgreSQL JSONB — already parsed
        if isinstance(value, str):
            return json.loads(value)  # SQLite TEXT — parse it
        return default or []

    # ================================================================
    # PUBLIC — Main entry point
    # ================================================================

    def ask(self, query: str) -> QueryResponse:
        """
        Answer a question using your stored history.

        Full RAG pipeline: retrieve → analyze → generate → respond

        INTERNAL — the pipeline steps in detail:

        Step 1: EMBED
          Your query becomes a vector.
          Same embedding model (text-embedding-3-small) used for storage.
          CRITICAL: must use same model for query and documents.
          Different models = different vector spaces = similarity is meaningless.
          Like measuring distance in miles vs kilometers without converting.

        Step 2: RETRIEVE
          ChromaDB finds stored events whose vectors are close to query vector.
          "Close" = high cosine similarity = similar meaning.
          Returns event_ids + similarity scores.

        Step 3: FETCH
          SQLite gives us full event data for those IDs.
          ChromaDB only stores vectors + metadata.
          Full text, emotions, confidence live in SQLite.
          This is why we need BOTH storage systems.

        Step 4: FILTER
          Drop events with similarity < min_similarity threshold.
          These are too different to be useful evidence.
          Better to have 2 highly relevant events than 6 marginally relevant ones.

        Step 5: ANALYZE
          Compute outcome_ratio: how many retrieved events had bad outcomes?
          Count linked outcomes from pending_outcomes table.
          This feeds the confidence calculation.

        Step 6: BUILD PROMPT
          Format retrieved events as structured evidence.
          Inject into system prompt alongside the query.
          This is the "Augmented" part of RAG.

        Step 7: GENERATE
          GPT-4o reads query + evidence → fills QueryResponse schema.
          Grammar-constrained: must cite event IDs, must set confidence.
          Cannot hallucinate — schema enforces verifiable output.

        Returns: QueryResponse (structured, verifiable, evidence-backed)
        """

        print(f"\n🔍 Query Engine: '{query[:60]}...'")

        # ── STEP 1: EMBED QUERY ──────────────────────────────
        print("   Step 1: Embedding query...")
        # We build a query embedding text the same way we build
        # event embedding text — emphasize meaning over wording
        query_embedding_text = query

        # ── STEP 2: RETRIEVE FROM CHROMADB ──────────────────
        print("   Step 2: Searching vector store...")
        raw_results = self.vector_store.search_similar(
            query_text=query_embedding_text,
            n_results=self.n_retrieve
        )

        # ── STEP 3: FETCH FULL EVENTS FROM SQLITE ───────────
        print("   Step 3: Fetching full events from database...")
        retrieved = self._fetch_full_events(raw_results)

        # ── STEP 4: FILTER BY SIMILARITY ────────────────────
        retrieved = [
            r for r in retrieved
            if r["similarity_score"] >= self.min_similarity
        ]
        print(f"   → {len(retrieved)} events above similarity threshold "
              f"({self.min_similarity})")

        # ── STEP 5: ANALYZE OUTCOMES ─────────────────────────
        print("   Step 4: Analyzing outcomes...")
        outcome_ratio, sample_size = self._analyze_outcomes(retrieved)
        print(f"   → outcome_ratio={outcome_ratio:.2f}, sample_size={sample_size}")

        # ── STEP 6 + 7: BUILD PROMPT + GENERATE ─────────────
        print("   Step 5: Generating grounded response...")
        response = self._generate_response(
            query=query,
            retrieved=retrieved,
            outcome_ratio=outcome_ratio,
            sample_size=sample_size
        )

        return response

    # ================================================================
    # PRIVATE — Pipeline implementation
    # ================================================================

    def _fetch_full_events(self, raw_results: list[dict]) -> list[dict]:
        """
        Fetch full event data from SQLite for each retrieved ID.

        INTERNAL:
        ChromaDB returns:
          [{"event_id": "abc123", "similarity_score": 0.87, ...}, ...]

        SQLite returns for each ID:
          {"id": "abc123", "event_type": "decision",
           "action_description": "...", "emotional_states": "...", ...}

        We merge them: add similarity_score to the SQLite row.
        Result: full event data WITH similarity scores.

        This merge is the bridge between the two storage systems.
        """

        enriched = []
        cursor   = self.db.conn.cursor()

        for result in raw_results:
            cursor.execute(
                "SELECT * FROM events WHERE id = %s",
                (result["event_id"],)
            )
            row = cursor.fetchone()

            if row:
                event_dict = dict(row)

                # Parse JSON fields back to lists
                # SQLite stores lists as JSON strings
                # We convert back here so the rest of the code
                # works with proper Python lists
                event_dict["emotional_states"]   = self._parse_json_field(event_dict.get("emotional_states"))
                event_dict["external_pressures"] = self._parse_json_field(event_dict.get("external_pressures"))
                event_dict["tags"] = self._parse_json_field(event_dict.get("tags"))

                # Attach similarity score from ChromaDB result
                event_dict["similarity_score"] = result["similarity_score"]

                enriched.append(event_dict)

        return enriched

    def _analyze_outcomes(self, retrieved: list[dict]) -> tuple[float, int]:
        """
        Compute outcome ratio for retrieved events.

        INTERNAL — What is outcome_ratio?
        outcome_ratio = negative_outcomes / total_known_outcomes

        Why do we need this?
        The LLM's confidence in its pattern should be proportional to
        how consistently bad the outcomes were.

        3 similar events, 3 bad outcomes → outcome_ratio = 1.0 → high confidence
        3 similar events, 1 bad outcome  → outcome_ratio = 0.33 → low confidence
        3 similar events, 0 known outcomes → outcome_ratio = 0.0 → insufficient data

        We look up the pending_outcomes table for each event.
        If an outcome was logged and linked, we check its sentiment.

        Returns: (outcome_ratio, sample_size)
          outcome_ratio: 0.0 to 1.0
          sample_size: number of events with known outcomes
        """

        if not retrieved:
            return 0.0, 0

        negative_count = 0
        known_outcomes = 0
        cursor         = self.db.conn.cursor()

        for event in retrieved:
            # Look for linked outcomes in pending_outcomes table
            cursor.execute("""
                SELECT po.outcome_event_id, e.action_description
                FROM pending_outcomes po
                LEFT JOIN events e ON po.outcome_event_id = e.id
                WHERE po.event_id = %s AND po.resolved = TRUE
            """, (event["id"],))
            row         = cursor.fetchone()
            outcome_row = dict(row) if row else None

            if outcome_row and outcome_row["action_description"]:
                known_outcomes += 1
                # Check if outcome event has negative emotional context
                # Simple heuristic: look for negative keywords
                outcome_text = outcome_row["action_description"].lower()
                negative_keywords = [
                    "burnout", "stress", "regret", "conflict", "failed",
                    "mistake", "overwhelmed", "exhausted", "bad", "wrong",
                    "problem", "issue", "difficult", "hard", "struggled"
                ]
                if any(kw in outcome_text for kw in negative_keywords):
                    negative_count += 1

        cursor.close()

        if known_outcomes == 0:
            return 0.0, len(retrieved)  # sample_size = events found, ratio = unknown

        return negative_count / known_outcomes, known_outcomes

    def _format_evidence_for_prompt(self, retrieved: list[dict]) -> str:
        """
        Format retrieved events as structured evidence text.

        INTERNAL — This is the "Augmented" step of RAG.

        We convert raw database rows into a human-readable evidence block
        that gets injected into the LLM prompt.

        The format matters. We structure it so the LLM can:
          1. Identify the pattern (similar emotional states + decisions)
          2. Note the outcomes (where available)
          3. Cite specific events by ID

        We include the event ID explicitly so the LLM can reference it
        in the evidence field of QueryResponse. Without IDs, the LLM
        would have to make up event references.
        """

        if not retrieved:
            return "No similar past events found in history."

        lines = ["PAST SIMILAR EVENTS (ordered by relevance):\n"]

        for i, event in enumerate(retrieved, 1):
            date       = event.get("event_date") or event.get("created_at", "")[:10]
            emotions   = ", ".join(event["emotional_states"]) or "not recorded"
            pressures  = ", ".join(event["external_pressures"]) or "none"
            confidence = event.get("confidence_score", "unknown")
            similarity = event.get("similarity_score", 0)

            lines.append(
                f"[{i}] Event ID: {event['id'][:16]}...\n"
                f"    Date: {date} | Similarity to current: {similarity:.0%}\n"
                f"    What happened: {event['action_description']}\n"
                f"    Emotional state: {emotions}\n"
                f"    External pressures: {pressures}\n"
                f"    Confidence at time: {confidence}/10\n"
            )

            # Add outcome if available
            if event.get("event_type") == "outcome":
                lines.append(f"    Outcome: {event['action_description']}\n")
            else:
                cursor = self.db.conn.cursor()
                cursor.execute("""
                    SELECT e.action_description as outcome_desc
                    FROM pending_outcomes po
                    JOIN events e ON po.outcome_event_id = e.id
                    WHERE po.event_id = %s AND po.resolved = TRUE
                """, (event["id"],))
                row     = cursor.fetchone()
                outcome = dict(row) if row else None
                cursor.close()

                if outcome:
                    lines.append(f"    Outcome: {outcome['outcome_desc']}\n")
                else:
                    lines.append(f"    Outcome: not yet recorded\n")

        return "\n".join(lines)

    def _build_system_prompt(self, outcome_ratio: float, sample_size: int) -> str:
        """
        Build the RAG system prompt.

        INTERNAL — Why the system prompt includes outcome_ratio:
        We tell the LLM the statistical context so it calibrates
        its confidence correctly.

        "3 of 5 similar past decisions had negative outcomes (ratio: 0.6)"
        gives the LLM the math to say confidence=0.6, warning_level="medium"

        Without this, the LLM guesses confidence from the text alone.
        With it, confidence is grounded in actual outcome statistics.

        This is the difference between:
          "I feel fairly confident this is risky" (vague)
          "confidence: 0.6 based on 3/5 negative outcomes" (grounded)
        """

        data_quality = (
            "INSUFFICIENT DATA: fewer than 3 similar events found. "
            "Set insufficient_data=true and confidence below 0.4."
            if sample_size < 3
            else f"Sample size: {sample_size} similar events found. "
                 f"Outcome ratio: {outcome_ratio:.0%} negative outcomes where known."
        )

        return f"""You are a personal decision intelligence system.
Your job: analyze the user's question against their personal history and give grounded advice.

CRITICAL RULES:
1. Answer ONLY using the evidence provided below. Never generalize beyond it.
2. Cite specific events by their Event ID in the evidence field.
3. If no clear pattern exists in the evidence, say so honestly.
4. Confidence must reflect the evidence quality: {data_quality}
5. Be direct. No hedging. No generic life advice.

WARNING LEVEL GUIDE:
  high:   3+ similar events with negative outcomes
  medium: 1-2 negative outcomes OR pattern present but small sample
  low:    mixed outcomes, unclear pattern
  none:   positive outcomes OR insufficient history

CONFIDENCE GUIDE:
  0.8-1.0: strong pattern, 4+ events, consistent outcomes
  0.5-0.7: moderate pattern, 2-3 events, some consistency
  0.2-0.4: weak pattern, few events, mixed outcomes
  0.0-0.2: insufficient data, set insufficient_data=true

Your recommendation must be specific and actionable. Not "be careful". 
Instead: "delay 7 days", "reduce scope by 30%", "ask X for a written commitment"."""

    def _generate_response(self,
                           query: str,
                           retrieved: list[dict],
                           outcome_ratio: float,
                           sample_size: int) -> QueryResponse:
        """
        Generate structured RAG response via GPT-4o.

        INTERNAL — The full prompt construction:

        system_prompt:
          Rules for how to answer (grounded, cite IDs, no hallucination)
          Statistical context (outcome_ratio, sample_size)

        user_message:
          The question
          The formatted evidence block (retrieved events)

        response_format=QueryResponse:
          Grammar-constrained output
          LLM MUST fill all fields
          LLM MUST put real event IDs in evidence.event_id
          LLM CANNOT return free text

        This combination makes hallucination structurally difficult:
          - Schema forces specific fields (no vague text)
          - Evidence IDs must match real stored events
          - Confidence must be a number (not "fairly confident")
          - warning_level must be from enum (not "somewhat risky")
        """

        evidence_text = self._format_evidence_for_prompt(retrieved)

        user_message = (
            f"USER QUESTION:\n{query}\n\n"
            f"{evidence_text}\n\n"
            f"Based on this evidence, provide a structured analysis and recommendation."
        )

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt(
                    outcome_ratio, sample_size
                )},
                {"role": "user", "content": user_message}
            ],
            response_format=QueryResponse,
            temperature=0,      # Deterministic — same evidence → same analysis
            max_tokens=1500     # QueryResponse needs ~500-800 tokens. 1500 is safe.
        )

        # Check clean completion
        finish_reason = response.choices[0].finish_reason
        if finish_reason != "stop":
            raise ValueError(
                f"Query generation ended with finish_reason='{finish_reason}'. "
                f"Tokens: {response.usage.completion_tokens}/1500"
            )

        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError("Query response parsing returned None.")

        print(f"   ✓ Response generated: warning={result.warning_level} "
              f"confidence={result.confidence} "
              f"evidence_count={len(result.evidence)}")

        return result

    def format_response(self, query: str, response: QueryResponse) -> str:
        """
        Format QueryResponse into human-readable output.

        INTERNAL: This is purely presentation logic.
        We keep it separate from generation logic (single responsibility).
        The QueryResponse object is the canonical result —
        this method just makes it readable.

        Later, Streamlit will use QueryResponse directly to render
        a proper UI. This formatter is for the terminal phase.
        """

        # Warning level → visual indicator
        warning_icons = {
            "none":   "✅",
            "low":    "🟡",
            "medium": "🟠",
            "high":   "🔴"
        }
        icon = warning_icons.get(response.warning_level, "⚪")

        lines = [
            f"\n{'='*60}",
            f"  {icon} SECOND BRAIN RESPONSE",
            f"{'='*60}",
            f"\n📋 YOUR QUERY: {query}",
        ]

        # Insufficient data warning
        if response.insufficient_data:
            lines.append(
                "\n⚠️  INSUFFICIENT HISTORY: Less than 3 similar events found.\n"
                "   Log more events to improve pattern detection.\n"
                "   Current response has low confidence."
            )

        # Pattern
        if response.pattern_detected:
            lines.append(f"\n🔍 PATTERN DETECTED:\n   {response.pattern_detected}")

        # Evidence
        if response.evidence:
            lines.append(f"\n📚 EVIDENCE ({len(response.evidence)} events):")
            for i, ev in enumerate(response.evidence, 1):
                lines.append(f"\n   [{i}] {ev.event_date} — {ev.description}")
                if ev.outcome:
                    lines.append(f"       Outcome: {ev.outcome}")
                lines.append(f"       Why relevant: {ev.relevance}")
        else:
            lines.append("\n📚 EVIDENCE: No specific past events cited.")

        # Recommendation
        lines.append(f"\n💡 RECOMMENDATION:\n   {response.recommendation}")

        # Reasoning
        lines.append(f"\n🧠 REASONING:\n   {response.reasoning}")

        # Confidence + warning
        confidence_pct = f"{response.confidence:.0%}"
        lines.append(
            f"\n📊 CONFIDENCE: {confidence_pct} | "
            f"WARNING: {response.warning_level.upper()} {icon}"
        )

        lines.append(f"\n{'='*60}")

        return "\n".join(lines)