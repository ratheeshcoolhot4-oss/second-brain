# app.py
#
# ============================================================
# STREAMLIT FRONTEND — SECOND BRAIN
# ============================================================
#
# WHAT IS STREAMLIT?
# ------------------
# Streamlit is a Python library that turns Python scripts
# into web apps. No HTML, no CSS, no JavaScript needed.
#
# HOW IT WORKS INTERNALLY:
# Every time the user interacts (clicks button, types text),
# Streamlit re-runs the entire Python script from top to bottom.
# This is called the "execution model".
#
# st.session_state persists data across reruns.
# Without it, every rerun starts fresh — all variables reset.
# We use session_state to keep the SecondBrain instance alive
# so we don't reconnect to DB/Qdrant on every button click.
#
# LAYOUT:
# ┌─────────────────────────────────────┐
# │  🧠 Second Brain                    │
# ├──────────┬──────────────────────────┤
# │ Log Entry│ Ask  │ Timeline │ People │
# │          │      │          │        │
# │ [input]  │[query│ events   │ people │
# │ [Log]    │ box] │ table    │ belief │
# │          │[Ask] │          │ vs ev  │
# │ result   │result│          │        │
# └──────────┴──────────────────────────┘
# ============================================================

import streamlit as st
import sys
import json
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()

# Support both local .env and Streamlit Cloud secrets
for key in ["OPENAI_API_KEY", "DATABASE_URL", "QDRANT_URL", "QDRANT_API_KEY"]:
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = st.secrets[key]

sys.path.insert(0, str(Path(__file__).parent))

from agents.archivist import ArchivistAgent
from agents.reasoning_graph import build_reasoning_graph
from models.schemas import GraphState
from storage.database import Database
from storage.vector_store import VectorStore

# ============================================================
# PAGE CONFIG — Must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title  = "Second Brain",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .warning-high   { background: #fff0f0; border-left: 4px solid #e74c3c; padding: 1rem; border-radius: 4px; }
    .warning-medium { background: #fff8f0; border-left: 4px solid #f39c12; padding: 1rem; border-radius: 4px; }
    .warning-low    { background: #fffff0; border-left: 4px solid #f1c40f; padding: 1rem; border-radius: 4px; }
    .warning-none   { background: #f0fff4; border-left: 4px solid #27ae60; padding: 1rem; border-radius: 4px; }
    .evidence-card  { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .belief-tag     { background: #fff3cd; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.85rem; }
    .evidence-tag   { background: #f8d7da; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.85rem; }
    .metric-card    { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZE SYSTEM — cached so it only runs once
#
# INTERNAL: @st.cache_resource caches the return value.
# On first run: connects to DB, Qdrant, builds graph (~3 seconds)
# On subsequent reruns: returns cached object instantly
# Without this: reconnects on every button click = very slow
# ============================================================

@st.cache_resource
def initialize_system():
    """Initialize all components once, cache for the session."""
    db           = Database()
    vector_store = VectorStore()
    archivist    = ArchivistAgent()
    graph, query_engine = build_reasoning_graph(
        db=db,
        vector_store=vector_store
    )
    return db, vector_store, archivist, graph, query_engine


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ingest_entry(raw_text: str, db, vector_store, archivist) -> dict:
    """Run the ingestion pipeline and return result."""
    event    = archivist.extract(raw_text)
    event_id = db.store_event(event)
    emb_id   = vector_store.store_event_vector(event_id, event)
    db.update_embedding_id(event_id, emb_id)
    return {"event_id": event_id, "event": event}


def run_ask(query: str, graph) -> dict:
    """Run the RAG reasoning graph and return state."""
    initial_state: GraphState = {
        "query":               query,
        "input_type":          "",
        "retrieved_events":    [],
        "similarity_scores":   [],
        "pattern":             None,
        "outcome_ratio":       0.0,
        "sample_size":         0,
        "response":            None,
        "final_output":        "",
        "error":               None,
        "pending_decision_id": None,
    }
    return graph.invoke(initial_state)


def warning_class(level: str) -> str:
    return {
        "high":   "warning-high",
        "medium": "warning-medium",
        "low":    "warning-low",
        "none":   "warning-none",
    }.get(level, "warning-none")


def warning_icon(level: str) -> str:
    return {"high": "🔴", "medium": "🟠", "low": "🟡", "none": "✅"}.get(level, "⚪")


# ============================================================
# MAIN APP
# ============================================================

def main():
    # ── HEADER ───────────────────────────────────────────────
    st.markdown('<div class="main-header">🧠 Second Brain</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personal Decision Intelligence System</div>', unsafe_allow_html=True)

    # ── LOAD SYSTEM ──────────────────────────────────────────
    try:
        db, vector_store, archivist, graph, query_engine = initialize_system()
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.info("Check your .env file for DATABASE_URL, QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY")
        return

    # ── STATS BAR ────────────────────────────────────────────
    stats     = db.get_stats()
    vec_stats = vector_store.get_collection_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events",    stats["table_counts"]["events"])
    with col2:
        st.metric("Vectors Stored",  vec_stats["total_vectors"])
    with col3:
        st.metric("People Tracked",  stats["table_counts"]["people_mentions"])
    with col4:
        st.metric("Pending Outcomes",stats["table_counts"]["pending_outcomes"])

    st.divider()

    # ── TABS ─────────────────────────────────────────────────
    tab_log, tab_ask, tab_timeline, tab_people, tab_outcomes = st.tabs([
        "📝 Log Entry",
        "🤔 Ask Your History",
        "📅 Timeline",
        "👥 People",
        "🔗 Outcomes"
    ])

    # ================================================================
    # TAB 1 — LOG ENTRY
    # ================================================================
    with tab_log:
        st.subheader("Log a Journal Entry")
        st.caption("Write anything — decisions, feelings, interactions, outcomes. The system extracts structure automatically.")

        # ── SESSION STATE KEY TRICK FOR CLEARING ─────────────
        # Streamlit widgets are identified by their key.
        # Changing the key forces Streamlit to create a fresh widget.
        # We increment a counter after each successful log.
        # New counter = new key = empty text box.
        if "log_counter" not in st.session_state:
            st.session_state.log_counter = 0

        entry_text = st.text_area(
            "Your entry",
            placeholder="e.g. Accepted extra project today. Feeling tired and under money pressure. Said yes even though something felt off.",
            height=120,
            key=f"log_input_{st.session_state.log_counter}"  # ← stable, resets on counter change
        )

        if st.button("Log Entry", type="primary", key="log_btn"):
            if not entry_text.strip():
                st.warning("Please write something first.")
            else:
                with st.spinner("Extracting structure..."):
                    try:
                        result = ingest_entry(entry_text, db, vector_store, archivist)
                        event  = result["event"]

                        # ── INCREMENT COUNTER → CLEARS TEXT BOX ──
                        st.session_state.log_counter += 1

                        st.success(f"✓ Logged as **{event.event_type.value.upper()}**")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**What happened:**")
                            st.info(event.action_description)

                            st.markdown("**Emotional state:**")
                            emotions = [e.value for e in event.emotional_states]
                            if emotions:
                                st.write(" · ".join([f"`{e}`" for e in emotions]))
                            else:
                                st.write("_none detected_")

                            st.markdown("**External pressures:**")
                            if event.external_pressures:
                                st.write(" · ".join([f"`{p}`" for p in event.external_pressures]))
                            else:
                                st.write("_none detected_")

                        with col_b:
                            st.markdown("**Confidence inferred:**")
                            confidence_color = (
                                "🔴" if event.confidence_score <= 3 else
                                "🟡" if event.confidence_score <= 6 else "🟢"
                            )
                            st.write(f"{confidence_color} **{event.confidence_score}/10** — _{event.confidence_basis}_")

                            if event.decision_pressure:
                                st.markdown("**Pressure source:**")
                                st.write(f"`{event.decision_pressure}`")

                            if event.tags:
                                st.markdown("**Tags:**")
                                st.write(" ".join([f"`{t}`" for t in event.tags]))

                        if event.people:
                            st.markdown("---")
                            st.markdown(f"**👥 {len(event.people)} person(s) detected:**")
                            for person in event.people:
                                with st.expander(f"🔵 {person.name}"):
                                    if person.stated_belief:
                                        st.markdown(f'<span class="belief-tag">💭 Belief: {person.stated_belief}</span>', unsafe_allow_html=True)
                                    if person.interaction_description:
                                        st.markdown(f'<span class="evidence-tag">📋 Evidence: {person.interaction_description}</span>', unsafe_allow_html=True)

                        st.cache_data.clear()

                    except Exception as e:
                        st.error(f"Error: {e}")
    # ================================================================
    # TAB 2 — ASK YOUR HISTORY
    # ================================================================
    with tab_ask:
        st.subheader("Ask Your History")
        st.caption("Ask anything about your past decisions. The system searches your history and reasons over evidence.")

        query_text = st.text_input(
            "Your question",
            placeholder="e.g. Should I take play today",
            key="ask_input"
        )

        if st.button("Ask", type="primary", key="ask_btn"):
            if not query_text.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching history and reasoning..."):
                    try:
                        result   = run_ask(query_text, graph)
                        response = result.get("response")

                        if response is None:
                            st.warning("No response generated. Try logging more events first.")
                        else:
                            # ── WARNING BANNER ────────────────────────
                            icon  = warning_icon(response.warning_level)
                            wclass = warning_class(response.warning_level)
                            st.markdown(
                                f'<div class="{wclass}">'
                                f'<strong>{icon} WARNING LEVEL: {response.warning_level.upper()}</strong> &nbsp;&nbsp; '
                                f'Confidence: <strong>{response.confidence:.0%}</strong>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown("")

                            # ── INSUFFICIENT DATA WARNING ──────────────
                            if response.insufficient_data:
                                st.warning("⚠️ Insufficient history — fewer than 3 similar events found. Log more entries to improve accuracy.")

                            # ── PATTERN ───────────────────────────────
                            if response.pattern_detected:
                                st.markdown("### 🔍 Pattern Detected")
                                st.info(response.pattern_detected)

                            # ── EVIDENCE ──────────────────────────────
                            if response.evidence:
                                st.markdown(f"### 📚 Evidence ({len(response.evidence)} events)")
                                for i, ev in enumerate(response.evidence, 1):
                                    with st.container():
                                        st.markdown(
                                            f'<div class="evidence-card">'
                                            f'<strong>[{i}] {ev.event_date}</strong> — {ev.description}'
                                            + (f'<br><em>Outcome: {ev.outcome}</em>' if ev.outcome else '')
                                            + f'<br><small>Why relevant: {ev.relevance}</small>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )

                            # ── RECOMMENDATION ────────────────────────
                            st.markdown("### 💡 Recommendation")
                            st.success(response.recommendation)

                            # ── REASONING ─────────────────────────────
                            with st.expander("🧠 Show reasoning"):
                                st.write(response.reasoning)

                    except Exception as e:
                        st.error(f"Error: {e}")

    # ================================================================
    # TAB 3 — TIMELINE
    # ================================================================
    with tab_timeline:
        st.subheader("Event Timeline")
        st.caption("Your logged history, most recent first.")

        # Filter controls
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            type_filter = st.selectbox(
                "Filter by type",
                ["All", "decision", "action", "emotion", "outcome", "interaction", "thought"],
                key="timeline_type_filter"
            )
        with col_f2:
            limit = st.slider("Show last N events", 10, 100, 30, key="timeline_limit")

        @st.cache_data(ttl=30)
        def get_events(lim):
            return db.get_recent_events(limit=lim)

        events = get_events(limit)

        # Apply type filter
        if type_filter != "All":
            events = [e for e in events if e["event_type"] == type_filter]

        if not events:
            st.info("No events logged yet. Use the 'Log Entry' tab to start.")
        else:
            for event in events:
                # Parse JSON fields if they come back as strings
                emotions = event.get("emotional_states", [])
                if isinstance(emotions, str):
                    emotions = json.loads(emotions)

                date = (event.get("event_date") or
                        event.get("created_at", "")[:10])

                type_emoji = {
                    "decision":    "🔵",
                    "action":      "🟢",
                    "emotion":     "🟡",
                    "outcome":     "🔴",
                    "interaction": "🟣",
                    "thought":     "⚪"
                }.get(event["event_type"], "⚫")

                with st.expander(
                    f"{type_emoji} **{date}** — {event['action_description'][:80]}"
                ):
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.write(f"**Type:** `{event['event_type']}`")
                        st.write(f"**Emotions:** {', '.join(emotions) if emotions else 'none'}")
                        if event.get("confidence_score"):
                            st.write(f"**Confidence:** {event['confidence_score']}/10")
                    with col_t2:
                        if event.get("decision_context"):
                            st.write(f"**Context:** {event['decision_context']}")
                        if event.get("embedding_summary"):
                            st.caption(f"Pattern: _{event['embedding_summary']}_")

    # ================================================================
    # TAB 4 — PEOPLE
    # ================================================================
    with tab_people:
        st.subheader("People Intelligence")
        st.caption("Belief stream vs Evidence stream per person. This is where contradiction detection lives.")

        people = db.get_all_people()

        if not people:
            st.info("No people tracked yet. Mention someone in a journal entry.")
        else:
            selected_person = st.selectbox(
                "Select person",
                people,
                key="people_selector"
            )

            if selected_person:
                history = db.get_person_history(selected_person)

                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Total Mentions", history["total_mentions"])
                with col_p2:
                    st.metric("Beliefs Stated", len(history["beliefs"]))
                with col_p3:
                    st.metric("Interactions Recorded", len(history["evidence"]))

                st.markdown("---")

                col_belief, col_evidence = st.columns(2)

                with col_belief:
                    st.markdown("### 💭 Belief Stream")
                    st.caption("What you SAID or THOUGHT about this person")
                    if history["beliefs"]:
                        for b in history["beliefs"]:
                            sentiment_icon = (
                                "🟢" if b["sentiment"] == "positive" else
                                "🔴" if b["sentiment"] == "negative" else "🟡"
                            )
                            st.markdown(
                                f'<div class="evidence-card">'
                                f'{sentiment_icon} <em>"{b["text"]}"</em>'
                                f'<br><small>{b["date"][:10]}</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No beliefs recorded yet.")

                with col_evidence:
                    st.markdown("### 📋 Evidence Stream")
                    st.caption("What actually HAPPENED with this person")
                    if history["evidence"]:
                        for e in history["evidence"]:
                            sentiment_icon = (
                                "🟢" if e["sentiment"] == "positive" else
                                "🔴" if e["sentiment"] == "negative" else "🟡"
                            )
                            st.markdown(
                                f'<div class="evidence-card">'
                                f'{sentiment_icon} <em>"{e["text"]}"</em>'
                                f'<br><small>{e["date"][:10]}</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No behavioral evidence recorded yet.")

                # ── CONTRADICTION CHECK ───────────────────────
                if history["beliefs"] and history["evidence"]:
                    positive_beliefs  = sum(1 for b in history["beliefs"]  if b["sentiment"] == "positive")
                    negative_evidence = sum(1 for e in history["evidence"] if e["sentiment"] == "negative")

                    if positive_beliefs > 0 and negative_evidence > 0:
                        st.markdown("---")
                        st.markdown(
                            f'<div class="warning-high">'
                            f'⚠️ <strong>POTENTIAL CONTRADICTION DETECTED</strong><br>'
                            f'You have <strong>{positive_beliefs} positive belief(s)</strong> about {selected_person} '
                            f'but <strong>{negative_evidence} negative interaction(s)</strong> recorded.<br>'
                            f'Your stated beliefs may not match the behavioral evidence.'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # ================================================================
    # TAB 5 — OUTCOMES
    # ================================================================
    with tab_outcomes:
        st.subheader("Pending Outcomes")
        st.caption("Decisions that don't have outcomes logged yet.")

        pending = db.get_pending_outcomes()

        if not pending:
            st.success("✓ All decisions have outcomes logged.")
        else:
            st.info(f"{len(pending)} decision(s) waiting for outcomes.")

            for i, p in enumerate(pending, 1):
                date = p.get("decision_date", p.get("created_at", ""))[:10]
                with st.expander(f"[{i}] {date} — {p['action_description']}"):
                    outcome_input = st.text_input(
                        "What happened?",
                        placeholder="e.g. Led to burnout after 3 weeks",
                        key=f"outcome_input_{i}"
                    )
                    if st.button("Log Outcome", key=f"outcome_btn_{i}"):
                        if outcome_input.strip():
                            with st.spinner("Storing outcome..."):
                                try:
                                    # Ingest outcome as event
                                    outcome_result = ingest_entry(
                                        f"Outcome: {outcome_input}",
                                        db, vector_store, archivist
                                    )
                                    # Link to decision
                                    db.resolve_pending_outcome(
                                        decision_event_id = p["event_id"],
                                        outcome_event_id  = outcome_result["event_id"]
                                    )
                                    st.success("✓ Outcome logged and linked to decision.")
                                    st.cache_data.clear()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.warning("Please describe what happened.")


if __name__ == "__main__":
    main()