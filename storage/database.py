# storage/database.py
#
# ============================================================
# POSTGRESQL — MIGRATED FROM SQLITE
# ============================================================
#
# WHAT CHANGED FROM SQLITE:
# --------------------------
# 1. Connection: sqlite3.connect(file) → psycopg2.connect(url)
# 2. Placeholders: ? → %s  (PostgreSQL standard)
# 3. Auto-increment: INTEGER PRIMARY KEY → SERIAL PRIMARY KEY
#    (PostgreSQL uses SERIAL, not SQLite's auto-increment)
# 4. JSON columns: TEXT storing JSON → JSONB
#    JSONB = binary JSON in PostgreSQL
#    Faster queries, indexable, validates JSON on insert
# 5. Boolean: INTEGER (0/1) → BOOLEAN (true/false)
# 6. commit(): same concept, slightly different handling
#
# WHAT STAYED THE SAME:
# ----------------------
# All method signatures are identical.
# store_event(), get_person_history(), get_pending_outcomes() etc.
# The rest of the app never knows storage changed.
# This is why we separated the storage layer.
#
# INTERNAL — WHY PSYCOPG2?
# ------------------------
# psycopg2 is the standard PostgreSQL adapter for Python.
# It speaks the PostgreSQL wire protocol directly.
# psycopg2-binary includes the C extension pre-compiled —
# no need to install PostgreSQL locally, just the Python package.
#
# CONNECTION POOLING NOTE:
# For a demo/personal app, one persistent connection is fine.
# Production would use a connection pool (psycopg2.pool or pgbouncer)
# to handle multiple concurrent users efficiently.
# ============================================================

import psycopg2
import psycopg2.extras
import json
import uuid
from datetime import datetime
from models.schemas import ExtractedEvent, ContradictionRecord
from dotenv import load_dotenv
import os

load_dotenv()


class Database:
    """
    PostgreSQL storage for all structured event data.
    Drop-in replacement for the SQLite version.
    All method signatures identical — rest of app unchanged.
    """

    def __init__(self):
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not found in .env")

        # ── CONNECT ───────────────────────────────────────────
        # psycopg2.connect() opens a persistent TCP connection
        # to Supabase's PostgreSQL server.
        #
        # cursor_factory=RealDictCursor means every row comes back
        # as a dict (like SQLite's row_factory = sqlite3.Row).
        # Without this, rows are plain tuples — harder to work with.
        self.conn = psycopg2.connect(
            database_url,
            cursor_factory=psycopg2.extras.RealDictCursor
        )

        # ── AUTOCOMMIT OFF (default) ──────────────────────────
        # PostgreSQL wraps everything in transactions by default.
        # We call self.conn.commit() explicitly after writes.
        # This gives us atomic operations — all inserts succeed
        # or all fail together. Same as SQLite behavior.
        self.conn.autocommit = False

        self._initialize_schema()
        print(f"📁 Database connected: Supabase PostgreSQL")

    def _initialize_schema(self):
        """
        Create all tables if they don't exist.

        POSTGRESQL DIFFERENCES FROM SQLITE:
        - JSONB instead of TEXT for JSON columns
          JSONB stores binary JSON — faster reads, supports indexing
          TEXT stored JSON as plain string — no validation, slower
        - BOOLEAN instead of INTEGER for true/false
        - TEXT stays TEXT (PostgreSQL has proper TEXT type)
        - SERIAL for auto-generated IDs (we use UUID so not needed here)
        - ON CONFLICT DO NOTHING for safe re-runs
        """

        cursor = self.conn.cursor()

        # ── EVENTS TABLE ──────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id                    TEXT PRIMARY KEY,
                event_type            TEXT NOT NULL,
                action_description    TEXT NOT NULL,
                raw_text              TEXT NOT NULL,
                emotional_states      JSONB DEFAULT '[]',
                external_pressures    JSONB DEFAULT '[]',
                decision_context      TEXT,
                decision_alternatives JSONB DEFAULT '[]',
                confidence_score      INTEGER,
                relates_to_decision   TEXT,
                event_date            TEXT,
                tags                  JSONB DEFAULT '[]',
                embedding_id          TEXT,
                embedding_summary     TEXT,
                created_at            TEXT NOT NULL
            )
        """)

        # ── PEOPLE MENTIONS TABLE ─────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people_mentions (
                id                      TEXT PRIMARY KEY,
                event_id                TEXT NOT NULL REFERENCES events(id),
                person_name             TEXT NOT NULL,
                relation_type           TEXT,
                stated_belief           TEXT,
                belief_sentiment        TEXT,
                interaction_description TEXT,
                interaction_sentiment   TEXT,
                created_at              TEXT NOT NULL
            )
        """)

        # ── CONTRADICTIONS TABLE ──────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id                 TEXT PRIMARY KEY,
                person_name        TEXT NOT NULL,
                belief_statement   TEXT NOT NULL,
                belief_date        TEXT NOT NULL,
                evidence_summary   TEXT NOT NULL,
                evidence_event_ids JSONB DEFAULT '[]',
                severity           TEXT,
                detected_at        TEXT NOT NULL
            )
        """)

        # ── PENDING OUTCOMES TABLE ────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_outcomes (
                id               TEXT PRIMARY KEY,
                event_id         TEXT NOT NULL REFERENCES events(id),
                description      TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                resolved         BOOLEAN DEFAULT FALSE,
                outcome_event_id TEXT,
                resolved_at      TEXT
            )
        """)

        # ── INDEXES ───────────────────────────────────────────
        # PostgreSQL indexes work the same as SQLite conceptually.
        # CREATE INDEX IF NOT EXISTS is safe to run repeatedly.
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_people_name ON people_mentions(person_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_people_event ON people_mentions(event_id)")

        self.conn.commit()
        print("   ✓ Schema initialized")

    def store_event(self, event: ExtractedEvent) -> str:
        """Store an extracted event. Returns the generated event ID."""

        event_id = str(uuid.uuid4())
        now      = datetime.now().isoformat()
        cursor   = self.conn.cursor()

        # ── INSERT MAIN EVENT ─────────────────────────────────
        # POSTGRESQL NOTE: %s placeholders, not ?
        # json.dumps() converts Python lists to JSON strings
        # PostgreSQL JSONB column accepts JSON strings directly
        cursor.execute("""
            INSERT INTO events (
                id, event_type, action_description, raw_text,
                emotional_states, external_pressures,
                decision_context, decision_alternatives,
                confidence_score, relates_to_decision,
                event_date, tags, embedding_id,
                embedding_summary, created_at
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s
            )
        """, (
            event_id,
            event.event_type.value,
            event.action_description,
            event.raw_text,
            json.dumps([e.value for e in event.emotional_states]),
            json.dumps(event.external_pressures),
            event.decision_context,
            json.dumps(event.decision_alternatives),
            event.confidence_score,
            event.relates_to_decision,
            event.event_date,
            json.dumps(event.tags),
            None,
            event.embedding_summary,
            now
        ))

        # ── INSERT PEOPLE MENTIONS ────────────────────────────
        for person in event.people:
            cursor.execute("""
                INSERT INTO people_mentions (
                    id, event_id, person_name, relation_type,
                    stated_belief, belief_sentiment,
                    interaction_description, interaction_sentiment,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()),
                event_id,
                person.name,
                person.relation_type.value,
                person.stated_belief,
                person.belief_sentiment,
                person.interaction_description,
                person.interaction_sentiment,
                now
            ))

        # ── CREATE PENDING OUTCOME FOR DECISIONS ──────────────
        if event.event_type.value == "decision":
            cursor.execute("""
                INSERT INTO pending_outcomes (
                    id, event_id, description, created_at, resolved
                ) VALUES (%s, %s, %s, %s, FALSE)
            """, (
                str(uuid.uuid4()),
                event_id,
                event.action_description,
                now
            ))

        self.conn.commit()
        print(f"   💾 Stored event: {event_id[:8]}... [{event.event_type.value}]")
        return event_id

    def update_embedding_id(self, event_id: str, embedding_id: str):
        """Link a stored event to its Qdrant vector ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE events SET embedding_id = %s WHERE id = %s",
            (embedding_id, event_id)
        )
        self.conn.commit()

    def get_person_history(self, person_name: str) -> dict:
        """
        Get complete history for a person — both belief and evidence streams.
        Used by Critic Agent for contradiction detection.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                pm.*,
                e.created_at as event_date,
                e.event_type
            FROM people_mentions pm
            JOIN events e ON pm.event_id = e.id
            WHERE LOWER(pm.person_name) = LOWER(%s)
            ORDER BY e.created_at ASC
        """, (person_name,))

        rows = cursor.fetchall()

        beliefs  = []
        evidence = []

        for row in rows:
            if row["stated_belief"]:
                beliefs.append({
                    "text":     row["stated_belief"],
                    "sentiment":row["belief_sentiment"],
                    "date":     row["event_date"],
                    "event_id": row["event_id"]
                })
            if row["interaction_description"]:
                evidence.append({
                    "text":     row["interaction_description"],
                    "sentiment":row["interaction_sentiment"],
                    "date":     row["event_date"],
                    "event_id": row["event_id"]
                })

        return {
            "person":         person_name,
            "total_mentions": len(rows),
            "beliefs":        beliefs,
            "evidence":       evidence
        }

    def get_pending_outcomes(self) -> list[dict]:
        """Get all decisions that don't have outcomes yet."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT po.*, e.action_description, e.created_at as decision_date
            FROM pending_outcomes po
            JOIN events e ON po.event_id = e.id
            WHERE po.resolved = FALSE
            ORDER BY po.created_at ASC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Get most recent events for timeline display."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM events
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            r = dict(row)
            # JSONB comes back as Python objects already in psycopg2
            # but normalize just in case
            if isinstance(r.get("emotional_states"), str):
                r["emotional_states"] = json.loads(r["emotional_states"])
            if isinstance(r.get("external_pressures"), str):
                r["external_pressures"] = json.loads(r["external_pressures"])
            if isinstance(r.get("tags"), str):
                r["tags"] = json.loads(r["tags"])
            result.append(r)
        return result

    def get_all_people(self) -> list[str]:
        """Get all unique people names tracked in the system."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT person_name
            FROM people_mentions
            ORDER BY person_name ASC
        """)
        return [row["person_name"] for row in cursor.fetchall()]

    def get_events_by_emotion(self, emotion: str) -> list[dict]:
        """Find all events where a specific emotion was present."""
        cursor = self.conn.cursor()
        # JSONB contains operator @> checks if array contains value
        cursor.execute("""
            SELECT * FROM events
            WHERE emotional_states @> %s
            ORDER BY created_at DESC
        """, (json.dumps([emotion]),))
        return [dict(row) for row in cursor.fetchall()]

    def store_contradiction(self, contradiction: ContradictionRecord) -> str:
        """Store a detected contradiction record."""
        contradiction_id = str(uuid.uuid4())
        cursor           = self.conn.cursor()
        cursor.execute("""
            INSERT INTO contradictions (
                id, person_name, belief_statement, belief_date,
                evidence_summary, evidence_event_ids, severity, detected_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            contradiction_id,
            contradiction.person_name,
            contradiction.belief_statement,
            contradiction.belief_date,
            contradiction.evidence_summary,
            json.dumps(contradiction.evidence_event_ids),
            contradiction.severity,
            contradiction.detected_at.isoformat()
        ))
        self.conn.commit()
        return contradiction_id

    def resolve_pending_outcome(self, decision_event_id: str,
                                 outcome_event_id: str):
        """Mark a pending outcome as resolved and link the outcome event."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE pending_outcomes
            SET resolved         = TRUE,
                outcome_event_id = %s,
                resolved_at      = %s
            WHERE event_id = %s
        """, (outcome_event_id, datetime.now().isoformat(), decision_event_id))
        self.conn.commit()

    def get_stats(self) -> dict:
        """Quick summary of what's stored."""
        # Flush any pending transaction so we read the latest data.
        # Without this, PostgreSQL's MVCC gives us a stale snapshot.
        self.conn.commit()
        cursor = self.conn.cursor()

        counts = {}
        for table in ["events", "people_mentions", "contradictions"]:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            counts[table] = cursor.fetchone()["count"]

        # Pending outcomes — only count unresolved
        cursor.execute("SELECT COUNT(*) as count FROM pending_outcomes WHERE resolved = FALSE")
        counts["pending_outcomes"] = cursor.fetchone()["count"]

        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM events
            GROUP BY event_type
        """)
        event_types = {row["event_type"]: row["count"]
                       for row in cursor.fetchall()}

        return {"table_counts": counts, "event_types": event_types}