# agents/reasoning_graph.py
#
# ============================================================
# LANGGRAPH — THE REASONING GRAPH
# ============================================================
#
# WHAT IS LANGGRAPH REALLY?
# -------------------------
# LangGraph is a state machine where:
#   Nodes = Python functions that transform state
#   Edges = connections between nodes (can be conditional)
#   State = a shared dict that flows through every node
#
# It is NOT a new kind of AI.
# It is NOT magic orchestration.
# It IS a structured way to run multiple LLM calls
# with shared state and conditional branching.
#
# WHY NOT JUST CALL FUNCTIONS SEQUENTIALLY?
# ------------------------------------------
# You could write:
#   result1 = retrieve(query)
#   result2 = analyze(result1)
#   result3 = advise(result2)
#   return format(result3)
#
# This works for a linear pipeline. But breaks when you need:
#   → Conditional paths ("if outcome, go here; if question, go there")
#   → Loops ("if confidence too low, retrieve more events")
#   → Shared state visible to all steps
#   → Easy visualization of what ran and why
#   → Error recovery ("if node fails, go to error handler")
#
# LangGraph gives you all of this with a clean API.
#
# HOW LANGGRAPH COMPILES:
# -----------------------
# When you call graph.compile(), LangGraph:
#   1. Validates all node names referenced in edges exist
#   2. Validates the graph has a START and at least one END
#   3. Builds an execution plan (which nodes can run in parallel)
#   4. Returns a Runnable — an object with .invoke() method
#
# When you call compiled.invoke(initial_state):
#   1. Sets current_node = START
#   2. Calls edge function to determine next node
#   3. Calls that node function with current state
#   4. Merges returned dict into state
#   5. Repeat from step 2 until current_node = END
#
# STATE MERGING (important internals):
# When a node returns {"retrieved_events": [...]}
# LangGraph merges this into the full state dict.
# The node only needs to return the keys it changed.
# Other keys remain unchanged in state.
#
# THE GRAPH WE BUILD:
# -------------------
#
#   START
#     │
#     ▼
#   router_node          ← classify: question / outcome / log
#     │
#     ├── "question" ──→ retrieve_node   ← RAG: search similar events
#     │                       │
#     │                       ▼
#     │                  analyze_node    ← find patterns in retrieved events
#     │                       │
#     │                       ▼
#     │                  advise_node     ← generate recommendation
#     │                       │
#     ├── "outcome" ───→ outcome_node    ← link outcome to past decision
#     │                       │
#     └── "log" ──────→ END (handled by main.py, not graph)
#                             │
#                             ▼
#                        respond_node    ← format final output
#                             │
#                             ▼
#                           END
# ============================================================

from langgraph.graph import StateGraph, START, END
from models.schemas import GraphState, QueryResponse
from agents.query_engine import QueryEngine
from storage.database import Database
from storage.vector_store import VectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()


# ============================================================
# NODE FUNCTIONS
#
# INTERNAL — Node function contract:
#   Input:  GraphState (the full current state)
#   Output: dict (only the keys this node changed)
#
# LangGraph merges the returned dict into state.
# You NEVER return the full state — only your changes.
# This prevents nodes from accidentally wiping other nodes' work.
# ============================================================

def router_node(state: GraphState) -> dict:
    """
    Classifies the input and sets input_type in state.

    INTERNAL — Why a router?
    We have one input interface but three different pipelines:
      question → RAG retrieval + reasoning
      outcome  → link to past decision, update pending_outcomes
      log      → Archivist pipeline (already in main.py)

    Without a router, we'd need separate commands for everything.
    The router lets ONE input interface handle everything by
    understanding what the input IS before deciding what to do.

    HOW IT CLASSIFIES:
    Simple heuristic rules first (fast, cheap, no API call):
      Starts with "outcome:" or "result:" → outcome
      Contains "?" or starts with question words → question
      Otherwise → question (default, most common case)

    We could use an LLM to classify, but that's expensive for
    something simple enough to rule-match. Save the LLM calls
    for the actual reasoning work.
    """

    query      = state["query"].lower().strip()
    input_type = "question"  # default

    # Outcome signals — explicit prefixes or past-tense outcome language
    outcome_signals = [
        "outcome:", "result:", "happened:", "update:",
        "ended in", "resulted in", "led to", "caused"
    ]
    if any(query.startswith(s) or s in query for s in outcome_signals):
        input_type = "outcome"

    print(f"   🔀 Router: classified as '{input_type}'")

    return {"input_type": input_type}


def retrieve_node(state: GraphState, query_engine: QueryEngine) -> dict:
    """
    Retrieves similar past events from the vector store.

    INTERNAL — What this node does:
    Embeds the query and searches ChromaDB for similar past events.
    Fetches full event data from SQLite.
    Filters by similarity threshold.

    We reuse QueryEngine._fetch_full_events() and the vector store
    directly here rather than calling query_engine.ask() because:
      1. We want each step separately in state for debugging
      2. analyze_node needs raw retrieved events, not a final response
      3. LangGraph's value is separating steps — collapsing them
         defeats the purpose

    State changes:
      retrieved_events:  list of full event dicts
      similarity_scores: parallel list of float scores
    """

    print(f"   📚 Retrieve node: searching for similar events...")

    raw_results = query_engine.vector_store.search_similar(
        query_text=state["query"],
        n_results=query_engine.n_retrieve
    )

    # Fetch full events from SQLite
    retrieved = query_engine._fetch_full_events(raw_results)

    # Filter by minimum similarity
    retrieved = [
        r for r in retrieved
        if r.get("similarity_score", 0) >= query_engine.min_similarity
    ]

    similarity_scores = [r.get("similarity_score", 0) for r in retrieved]

    print(f"   → Retrieved {len(retrieved)} events above threshold")

    return {
        "retrieved_events":  retrieved,
        "similarity_scores": similarity_scores,
    }


def analyze_node(state: GraphState, query_engine: QueryEngine) -> dict:
    """
    Analyzes retrieved events to find patterns.

    INTERNAL — What "analysis" means here:
    We compute quantitative signals that feed the LLM's reasoning:

    1. outcome_ratio: what fraction of similar past decisions had bad outcomes?
       High ratio → strong pattern → high confidence warning
       Low ratio  → unclear pattern → low confidence

    2. sample_size: how many similar events did we find?
       High sample → reliable pattern → higher confidence
       Low sample  → insufficient data → lower confidence

    3. pattern (optional pre-analysis): do the retrieved events share
       a common emotional state or context? We note this for the LLM.

    WHY COMPUTE THIS SEPARATELY FROM LLM REASONING?
    Because these are DETERMINISTIC calculations over structured data.
    We don't want the LLM estimating outcome ratios from text.
    We compute the math, give it to the LLM as facts.
    LLM does qualitative reasoning. We do quantitative computation.
    Clean separation of what each is good at.
    """

    print(f"   🔬 Analyze node: computing patterns...")

    retrieved    = state.get("retrieved_events", [])
    outcome_ratio, sample_size = query_engine._analyze_outcomes(retrieved)

    # Pre-compute dominant emotional pattern for the LLM
    all_emotions = []
    for event in retrieved:
        all_emotions.extend(event.get("emotional_states", []))

    pattern = None
    if all_emotions:
        # Find most common emotion across retrieved events
        from collections import Counter
        emotion_counts = Counter(all_emotions)
        most_common    = emotion_counts.most_common(2)
        if most_common:
            dominant = [e[0] for e in most_common]
            pattern  = f"decisions made while {' and '.join(dominant)}"

    print(f"   → outcome_ratio={outcome_ratio:.2f}, "
          f"sample_size={sample_size}, pattern='{pattern}'")

    return {
        "pattern":       pattern,
        "outcome_ratio": outcome_ratio,
        "sample_size":   sample_size,
    }


def advise_node(state: GraphState, query_engine: QueryEngine) -> dict:
    """
    Generates the structured recommendation using RAG.

    INTERNAL — This is where the LLM reasoning happens.
    All previous nodes prepared the context:
      retrieve_node → retrieved_events (the evidence)
      analyze_node  → outcome_ratio, sample_size (the statistics)

    This node takes all of that and generates a QueryResponse.

    The LLM call here is different from extraction:
      Extraction: LLM reads text → fills structured schema
      Advice:     LLM reads evidence → reasons → fills structured schema

    Same mechanism (grammar-constrained output) but different cognitive task.
    Extraction is pattern matching. Advice is reasoning over evidence.
    """

    print(f"   💡 Advise node: generating recommendation...")

    response = query_engine._generate_response(
        query        = state["query"],
        retrieved    = state.get("retrieved_events", []),
        outcome_ratio= state.get("outcome_ratio", 0.0),
        sample_size  = state.get("sample_size",   0)
    )

    return {"response": response}


def outcome_node(state: GraphState, db: Database) -> dict:
    """
    Links a logged outcome to its original decision.

    INTERNAL — The outcome linking problem:
    You make a decision on Mar 3.
    Burnout happens Apr 28 (8 weeks later).
    How does the system connect these?

    Phase 1 created pending_outcomes rows for every decision.
    This node:
      1. Takes the outcome text from the query
      2. Searches pending_outcomes for matching unresolved decisions
      3. Finds the best match using simple text similarity
      4. Marks it resolved, links the outcome

    WHY NOT USE VECTOR SEARCH HERE?
    Pending outcomes are few (you make ~1 decision per day).
    Full vector search is expensive for small sets.
    Simple string matching on action_description works well enough.
    Use the right tool for the scale of the problem.
    """

    print(f"   🔗 Outcome node: linking outcome to decision...")

    outcome_text = state["query"]

    # Clean the outcome prefix if present
    for prefix in ["outcome:", "result:", "happened:", "update:"]:
        if outcome_text.lower().startswith(prefix):
            outcome_text = outcome_text[len(prefix):].strip()
            break

    # Get unresolved pending outcomes
    pending = db.get_pending_outcomes()

    if not pending:
        return {
            "final_output": (
                "⚠️  No pending decisions found to link this outcome to.\n"
                "   Log a decision first, then log its outcome."
            )
        }

    # Find best match using simple word overlap
    # Simple but effective for small sets
    outcome_words = set(outcome_text.lower().split())
    best_match    = None
    best_score    = 0

    for pending_decision in pending:
        decision_words = set(pending_decision["action_description"].lower().split())
        overlap        = len(outcome_words & decision_words)
        if overlap > best_score:
            best_score = overlap
            best_match = pending_decision

    if not best_match or best_score < 2:
        # Show pending decisions so user can manually match
        pending_list = "\n".join([
            f"  [{i+1}] {p['action_description']} ({p['decision_date'][:10]})"
            for i, p in enumerate(pending[:5])
        ])
        return {
            "final_output": (
                f"⚠️  Could not automatically match outcome to a decision.\n"
                f"   Your pending decisions:\n{pending_list}\n"
                f"   Use /link <decision_number> <outcome_text> to link manually."
            )
        }

    # Store the outcome as a new event first
    # (handled by main.py ingest — we just return the match info here)
    return {
        "final_output": (
            f"🔗 OUTCOME LINKED:\n"
            f"   Decision: {best_match['action_description']}\n"
            f"   Made on:  {best_match['decision_date'][:10]}\n"
            f"   Outcome:  {outcome_text}\n\n"
            f"   ✓ This connection will improve future pattern detection.\n"
            f"   Pending decision ID: {best_match['event_id']}"
        ),
        # Store the pending_id so main.py can resolve it after ingesting the outcome
        "pending_decision_id": best_match["event_id"]
    }


def respond_node(state: GraphState, query_engine: QueryEngine) -> dict:
    """
    Formats the final output for display.

    INTERNAL — Why a dedicated respond node?
    Single responsibility. The advise_node produces a QueryResponse object.
    The respond_node turns that object into a human-readable string.
    Keeping them separate means:
      → We can change the display format without touching reasoning logic
      → Streamlit can use QueryResponse directly (skip this node)
      → Terminal uses formatted string (uses this node)
    """

    response = state.get("response")

    if response is None:
        return {"final_output": "⚠️  No response was generated."}

    formatted = query_engine.format_response(
        query    = state["query"],
        response = response
    )

    return {"final_output": formatted}


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_reasoning_graph(db: Database,
                          vector_store: VectorStore) -> object:
    """
    Assembles and compiles the LangGraph reasoning graph.

    INTERNAL — The compilation process:

    StateGraph(GraphState) creates a graph that uses GraphState
    as its state schema. LangGraph validates that all nodes
    return dicts with keys that exist in GraphState.

    add_node(name, function) registers a node.
    The function must accept GraphState and return dict.

    add_edge(from, to) adds an unconditional edge.
    After 'from' node runs, always go to 'to' node.

    add_conditional_edges(from, condition_fn, mapping) adds
    a conditional edge. condition_fn receives state and returns
    a string key. mapping maps that key to a node name.

    compile() validates and returns a Runnable.
    invoke(state) runs the graph from START to END.

    PARTIAL FUNCTIONS — Why we use functools.partial:
    Node functions need db and query_engine as dependencies.
    But LangGraph calls node functions with only (state,) as argument.
    partial() pre-fills the extra arguments:
      partial(retrieve_node, query_engine=qe)
      When LangGraph calls this with (state,), it becomes:
      retrieve_node(state, query_engine=qe)
    This is dependency injection without a framework.
    """

    from functools import partial

    query_engine = QueryEngine(db=db, vector_store=vector_store)

    # ── BUILD GRAPH ──────────────────────────────────────────
    graph = StateGraph(GraphState)

    # ── REGISTER NODES ───────────────────────────────────────
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", partial(retrieve_node, query_engine=query_engine))
    graph.add_node("analyze",  partial(analyze_node,  query_engine=query_engine))
    graph.add_node("advise",   partial(advise_node,   query_engine=query_engine))
    graph.add_node("outcome",  partial(outcome_node,  db=db))
    graph.add_node("respond",  partial(respond_node,  query_engine=query_engine))

    # ── EDGES FROM START ─────────────────────────────────────
    graph.add_edge(START, "router")

    # ── CONDITIONAL EDGE: router → different paths ───────────
    # After router runs, check state["input_type"] to decide next node
    def route_decision(state: GraphState) -> str:
        """
        INTERNAL — This function is called by LangGraph after router_node.
        It reads state and returns a string key.
        The key maps to a node name in the routing_map below.

        This is how LangGraph implements branching:
          Not if/else in code
          But graph edges with conditions
          The graph structure IS the control flow
        """
        return state.get("input_type", "question")

    graph.add_conditional_edges(
        "router",          # from this node
        route_decision,    # call this function to decide
        {                  # map return value → next node
            "question": "retrieve",
            "outcome":  "outcome",
            "log":      END,       # log inputs handled by main.py
        }
    )

    # ── LINEAR EDGES: question path ──────────────────────────
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze",  "advise")
    graph.add_edge("advise",   "respond")
    graph.add_edge("respond",   END)

    # ── LINEAR EDGES: outcome path ───────────────────────────
    graph.add_edge("outcome", END)

    # ── COMPILE ──────────────────────────────────────────────
    compiled = graph.compile()

    print("   ✓ Reasoning graph compiled")
    return compiled, query_engine


# ============================================================
# STANDALONE TEST
# Run: python agents/reasoning_graph.py
# Requires: events already stored in Phase 1
# ============================================================

if __name__ == "__main__":
    from storage.database import Database
    from storage.vector_store import VectorStore

    print("\n" + "="*60)
    print("  REASONING GRAPH — TEST")
    print("="*60)

    db           = Database()
    vector_store = VectorStore()
    graph, _     = build_reasoning_graph(db, vector_store)

    # Test question
    initial_state: GraphState = {
        "query":             "Should I take on this new project? Feeling exhausted.",
        "input_type":        "",
        "retrieved_events":  [],
        "similarity_scores": [],
        "pattern":           None,
        "outcome_ratio":     0.0,
        "sample_size":       0,
        "response":          None,
        "final_output":      "",
        "error":             None,
    }

    result = graph.invoke(initial_state)
    print(result["final_output"])