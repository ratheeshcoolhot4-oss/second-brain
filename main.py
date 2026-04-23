# main.py
#
# ============================================================
# SECOND BRAIN — PHASE 2
# ============================================================
#
# PHASE 1 (ingestion):
#   You type → extract → store → embed
#
# PHASE 2 (intelligence) — NOW ADDED:
#   /ask   → RAG query → pattern detection → recommendation
#   /outcome → link outcome to past decision
#
# OPTION B (current): explicit commands
#   /log     <text>   → ingest journal entry
#   /ask     <query>  → ask reasoning engine
#   /outcome <text>   → log outcome + link to decision
#   /search  <query>  → raw semantic search (debug)
#   /pending          → show unresolved decisions
#   /stats            → storage summary
#   /quit             → exit
#
# OPTION A (Phase 3): single input, router decides
#   Any text → Router classifies → correct pipeline runs
# ============================================================

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from agents.archivist import ArchivistAgent
from agents.reasoning_graph import build_reasoning_graph
from models.schemas import GraphState
from storage.database import Database
from storage.vector_store import VectorStore

console = Console()


class SecondBrain:
    """
    Main orchestrator. Wires all components together.

    INTERNAL — Why one class instead of loose functions?
    All components share the same db and vector_store connections.
    One class = one place to initialize shared resources.
    Methods = clean interface for each operation.
    """

    def __init__(self):
        console.print("\n[bold blue]🧠 Second Brain — Initializing...[/bold blue]\n")

        # ── SHARED RESOURCES (initialized once) ──────────────
        self.archivist    = ArchivistAgent()
        self.db           = Database()
        self.vector_store = VectorStore()

        # ── PHASE 2: REASONING GRAPH ──────────────────────────
        # build_reasoning_graph returns:
        #   compiled_graph: LangGraph Runnable
        #   query_engine:   QueryEngine instance (for format_response)
        #
        # INTERNAL: We build the graph once at startup.
        # Why not build it per-request?
        # graph.compile() validates the graph structure and builds
        # the execution plan. This takes ~50-100ms.
        # Building once at startup → every request uses the cached plan.
        console.print("\n[yellow]Building reasoning graph...[/yellow]")
        self.graph, self.query_engine = build_reasoning_graph(
            db=self.db,
            vector_store=self.vector_store
        )

        console.print("\n[bold green]✓ All systems ready[/bold green]\n")

    # ================================================================
    # /log — Ingest a journal entry (Phase 1 pipeline)
    # ================================================================

    def ingest(self, raw_text: str) -> dict:
        """
        Ingest pipeline: raw text → extract → store → embed.
        Same as Phase 1. Unchanged.
        """

        console.print(Panel(
            f"[italic]{raw_text}[/italic]",
            title="📝 Logging Entry",
            border_style="blue"
        ))

        # Extract
        console.print("\n[yellow]Extracting structure...[/yellow]")
        event = self.archivist.extract(raw_text)

        # Store in SQLite
        console.print("\n[yellow]Storing...[/yellow]")
        event_id = self.db.store_event(event)

        # Embed + store in ChromaDB
        console.print("\n[yellow]Embedding...[/yellow]")
        embedding_id = self.vector_store.store_event_vector(event_id, event)
        self.db.update_embedding_id(event_id, embedding_id)

        # Show what was understood
        self._display_extraction(event, event_id)

        return {"event_id": event_id, "event_type": event.event_type.value}

    # ================================================================
    # /ask — RAG query through reasoning graph (Phase 2)
    # ================================================================

    def ask(self, query: str):
        """
        Phase 2 RAG pipeline through LangGraph.

        INTERNAL — What invoke() does:
        graph.invoke(initial_state) runs the full graph:
          START → router → retrieve → analyze → advise → respond → END

        Each node reads from state, does its work, updates state.
        final_output in state contains the formatted response.

        initial_state must contain ALL keys defined in GraphState.
        Missing keys cause KeyError inside nodes.
        We initialize everything to safe defaults here.
        """

        console.print(f"\n[bold yellow]🤔 Asking: {query}[/bold yellow]\n")

        # ── INITIALIZE STATE ─────────────────────────────────
        # INTERNAL: Every key in GraphState must be present.
        # Nodes update only the keys they change.
        # LangGraph merges node output into this dict.
        initial_state: GraphState = {
            "query":             query,
            "input_type":        "",        # set by router_node
            "retrieved_events":  [],        # set by retrieve_node
            "similarity_scores": [],        # set by retrieve_node
            "pattern":           None,      # set by analyze_node
            "outcome_ratio":     0.0,       # set by analyze_node
            "sample_size":       0,         # set by analyze_node
            "response":          None,      # set by advise_node
            "final_output":      "",        # set by respond_node
            "error":             None,
            "pending_decision_id": None, 
        }

        # ── RUN GRAPH ────────────────────────────────────────
        result = self.graph.invoke(initial_state)

        # ── DISPLAY OUTPUT ───────────────────────────────────
        console.print(result["final_output"])

        # ── SHOW RETRIEVED EVENTS COUNT ──────────────────────
        n_retrieved = len(result.get("retrieved_events", []))
        if n_retrieved == 0:
            console.print(
                "\n[dim]💡 Tip: Log more events to improve pattern detection.[/dim]"
            )

    # ================================================================
    # /outcome — Log an outcome and link to past decision
    # ================================================================

    def log_outcome(self, outcome_text: str):
        """
        Log an outcome AND link it to the most likely past decision.

        Two-step process:
          Step 1: Run outcome through reasoning graph
                  (outcome_node finds the matching pending decision)
          Step 2: Ingest the outcome as a new event
                  (stores it + embeds it like any other entry)
          Step 3: Link in database (resolve pending_outcome)
        """

        console.print(f"\n[bold yellow]🔗 Logging outcome: {outcome_text}[/bold yellow]\n")

        # ── STEP 1: FIND MATCHING DECISION VIA GRAPH ─────────
        initial_state: GraphState = {
            "query":             f"outcome: {outcome_text}",
            "input_type":        "outcome",   # skip router, go direct to outcome_node
            "retrieved_events":  [],
            "similarity_scores": [],
            "pattern":           None,
            "outcome_ratio":     0.0,
            "sample_size":       0,
            "response":          None,
            "final_output":      "",
            "error":             None,
            "pending_decision_id": None, 
        }

        result = self.graph.invoke(initial_state)
        console.print(result["final_output"])

        # ── STEP 2: INGEST THE OUTCOME AS AN EVENT ────────────
        console.print("\n[yellow]Storing outcome as event...[/yellow]")
        outcome_entry = f"Outcome: {outcome_text}"
        ingest_result = self.ingest(outcome_entry)

        # ── STEP 3: LINK IN DATABASE ──────────────────────────
        pending_id = result.get("pending_decision_id")
        if pending_id and ingest_result.get("event_id"):
            cursor = self.db.conn.cursor()
            cursor.execute("""
                UPDATE pending_outcomes
                SET resolved = TRUE,
                    outcome_event_id = ?,
                    resolved_at = datetime('now')
                WHERE event_id = ?
            """, (ingest_result["event_id"], pending_id))
            self.db.conn.commit()
            console.print(
                f"\n[green]✓ Outcome linked to decision. "
                f"Pattern engine will use this connection.[/green]"
            )

    # ================================================================
    # /pending — Show unresolved decisions
    # ================================================================

    def show_pending(self):
        """
        Show decisions that don't have outcomes logged yet.

        INTERNAL: These are rows in pending_outcomes where resolved=0.
        Created automatically when any decision-type event is logged.
        Resolved when /outcome is used.
        """

        pending = self.db.get_pending_outcomes()

        if not pending:
            console.print("\n[green]✓ No pending decisions — all outcomes logged.[/green]")
            return

        console.print(f"\n[bold]⏳ Pending Decisions ({len(pending)}):[/bold]")
        console.print("[dim]These decisions don't have outcomes logged yet.[/dim]\n")

        for i, p in enumerate(pending, 1):
            date = p.get("decision_date", p.get("created_at", ""))[:10]
            console.print(f"  [{i}] {date} — {p['action_description']}")

        console.print(
            f"\n[dim]Use /outcome <what happened> to log an outcome.[/dim]"
        )

    # ================================================================
    # /search — Raw semantic search (debug/explore)
    # ================================================================

    def search(self, query: str, n: int = 5):
        """Raw semantic search. Shows similar events without reasoning."""

        console.print(f"\n[bold]🔍 Semantic Search: '{query}'[/bold]\n")
        results = self.vector_store.search_similar(query, n_results=n)

        if not results:
            console.print("  No results found.")
            return

        for result in results:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT * FROM events WHERE id = ?", (result["event_id"],))
            row = cursor.fetchone()
            if row:
                emotions = json.loads(row["emotional_states"] or "[]")
                console.print(
                    f"  {result['similarity_score']:.0%} | "
                    f"[{row['event_type']}] {row['action_description']}"
                )
                if emotions:
                    console.print(f"         emotions: {', '.join(emotions)}")

    # ================================================================
    # /stats — Storage summary
    # ================================================================

    def stats(self):
        db_stats  = self.db.get_stats()
        vec_stats = self.vector_store.get_collection_stats()

        console.print("\n[bold]📊 Second Brain Stats:[/bold]")
        console.print(f"  Total events:     {db_stats['table_counts']['events']}")
        console.print(f"  Total vectors:    {vec_stats['total_vectors']}")
        console.print(f"  People tracked:   {db_stats['table_counts']['people_mentions']}")
        console.print(f"  Pending outcomes: {db_stats['table_counts']['pending_outcomes']}")
        console.print(f"  Contradictions:   {db_stats['table_counts']['contradictions']}")
        console.print(f"\n  Event breakdown:")
        for event_type, count in db_stats["event_types"].items():
            console.print(f"    {event_type}: {count}")

    # ================================================================
    # Display helpers
    # ================================================================

    def _display_extraction(self, event, event_id: str):
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="bold cyan", width=22)
        table.add_column("Value", style="white")

        table.add_row("Event ID",    event_id[:16] + "...")
        table.add_row("Type",        event.event_type.value.upper())
        table.add_row("Description", event.action_description)
        table.add_row("Emotions",    ", ".join([e.value for e in event.emotional_states]) or "none")
        table.add_row("Pressures",   ", ".join(event.external_pressures) or "none")
        table.add_row("Confidence",  f"{event.confidence_score}/10 ({event.confidence_basis})")
        table.add_row("Pressure",    event.decision_pressure or "none")
        table.add_row("People",      str(len(event.people)) + " detected")

        console.print("\n")
        console.print(Panel(
            renderable=table,
            title="🤖 Extracted",
            border_style="green"
        ))

        if event.people:
            console.print("\n[bold]👥 People:[/bold]")
            for person in event.people:
                console.print(f"  {person.name}")
                if person.stated_belief:
                    console.print(f"    [yellow]Belief:[/yellow]   {person.stated_belief}")
                if person.interaction_description:
                    console.print(f"    [red]Evidence:[/red]  {person.interaction_description}")


# ============================================================
# INTERACTIVE SESSION
# ============================================================

def print_help():
    console.print("\n[bold]Commands:[/bold]")
    console.print("  <text>             → log journal entry")
    console.print("  /ask <query>       → ask reasoning engine (Phase 2)")
    console.print("  /outcome <text>    → log outcome + link to decision")
    console.print("  /pending           → show unresolved decisions")
    console.print("  /search <query>    → raw semantic search")
    console.print("  /stats             → storage summary")
    console.print("  /help              → show this")
    console.print("  /quit              → exit\n")


def main():
    brain = SecondBrain()
    print_help()

    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()

            if not user_input:
                continue
            elif user_input == "/quit":
                console.print("Goodbye.")
                break
            elif user_input == "/help":
                print_help()
            elif user_input == "/stats":
                brain.stats()
            elif user_input == "/pending":
                brain.show_pending()
            elif user_input.startswith("/ask "):
                query = user_input[5:].strip()
                if query:
                    brain.ask(query)
                else:
                    console.print("[red]Usage: /ask <your question>[/red]")
            elif user_input.startswith("/outcome "):
                outcome = user_input[9:].strip()
                if outcome:
                    brain.log_outcome(outcome)
                else:
                    console.print("[red]Usage: /outcome <what happened>[/red]")
            elif user_input.startswith("/search "):
                query = user_input[8:].strip()
                brain.search(query)
            elif user_input.startswith("/"):
                console.print(f"[red]Unknown command. Type /help for commands.[/red]")
            else:
                # No command prefix → treat as journal entry
                brain.ingest(user_input)

        except KeyboardInterrupt:
            console.print("\nGoodbye.")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise


if __name__ == "__main__":
    main()