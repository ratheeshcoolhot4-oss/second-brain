# storage/vector_store.py
#
# ============================================================
# QDRANT — MIGRATED FROM CHROMADB
# ============================================================
#
# WHAT CHANGED FROM CHROMADB:
# ----------------------------
# 1. Client: chromadb.PersistentClient(path) → QdrantClient(url, api_key)
#    ChromaDB stored data in a local folder.
#    Qdrant stores data on their cloud server.
#    Same concept, different location.
#
# 2. Collection creation: different API but same concept
#    ChromaDB: client.get_or_create_collection(name, metadata)
#    Qdrant:   client.recreate_collection(name, vectors_config)
#
# 3. Upsert: same concept, different method signature
#    ChromaDB: collection.upsert(ids, embeddings, documents, metadatas)
#    Qdrant:   client.upsert(collection_name, points=[PointStruct(...)])
#
# 4. Search: same concept, different return format
#    ChromaDB: collection.query(query_embeddings, n_results, where)
#    Qdrant:   client.search(collection_name, query_vector, limit, query_filter)
#
# WHAT STAYED THE SAME:
# ----------------------
# search_similar() signature: identical
# store_event_vector() signature: identical
# Return format: identical (list of dicts with event_id, similarity_score)
# The rest of the app never knows storage changed.
#
# INTERNAL — QDRANT CONCEPTS:
# ---------------------------
# Collection = ChromaDB collection = table in SQL
# Point = one stored vector + payload
#   - id:      unique identifier (we use event_id as string UUID)
#   - vector:  the embedding (list of 1536 floats)
#   - payload: metadata dict (event_type, emotional_states, etc.)
#
# PointStruct = Qdrant's object for a point to be inserted
# ScoredPoint = Qdrant's object returned from search
#   - id:      the point's ID
#   - score:   cosine similarity (already converted, not distance)
#   - payload: the metadata we stored
#
# IMPORTANT DIFFERENCE FROM CHROMADB:
# Qdrant search() returns SIMILARITY directly (higher = more similar)
# ChromaDB returned DISTANCE (lower = more similar)
# So we no longer need the "1 - distance" conversion.
# ============================================================

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from utils.embeddings import EmbeddingEngine
from models.schemas import ExtractedEvent
from datetime import datetime
from dotenv import load_dotenv
import os
import uuid

load_dotenv()


class VectorStore:
    """
    Qdrant cloud wrapper for semantic similarity search.
    Drop-in replacement for the ChromaDB version.
    All method signatures identical.
    """

    def __init__(self):
        qdrant_url     = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY not found in .env")

        # ── CONNECT TO QDRANT CLOUD ───────────────────────────
        # QdrantClient connects to the cloud cluster via HTTPS.
        # api_key authenticates your requests.
        # All vectors stored here persist across restarts —
        # no local folder, no data loss on redeploy.
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        self.collection_name   = "life_events"
        self.embedding_engine  = EmbeddingEngine()
        self.vector_dimensions = 1536  # text-embedding-3-small

        self._initialize_collection()
        print(f"🧠 Vector store connected: Qdrant Cloud")

    def _initialize_collection(self):
        """
        Create collection if it doesn't exist.

        INTERNAL — Qdrant collection config:

        VectorParams specifies:
          size:     dimensions of vectors we'll store (1536)
          distance: similarity metric to use

        Distance.COSINE:
          Same metric as ChromaDB "cosine" setting.
          Measures angle between vectors.
          Range: -1 to 1, higher = more similar.
          Perfect for semantic similarity.

        IMPORTANT: Unlike ChromaDB which needed "1 - distance" conversion,
        Qdrant.search() returns scores directly as cosine similarity.
        No conversion needed in our search_similar() method.
        """

        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimensions,
                    distance=Distance.COSINE
                )
            )
            print(f"   ✓ Created collection '{self.collection_name}'")
        else:
            count = self.client.count(self.collection_name).count
            print(f"   ✓ Collection '{self.collection_name}': {count} vectors stored")

    def store_event_vector(self, event_id: str, event: ExtractedEvent) -> str:
        """
        Embed an event and store it in Qdrant.
        Returns event_id (used as the point ID in Qdrant).
        """

        # ── BUILD EMBEDDING TEXT ──────────────────────────────
        embedding_text = self.embedding_engine.build_embedding_text(
            embedding_summary  = event.embedding_summary,
            emotional_states   = [e.value for e in event.emotional_states],
            external_pressures = event.external_pressures,
            people             = [p.name for p in event.people]
        )

        # ── GENERATE VECTOR ───────────────────────────────────
        vector = self.embedding_engine.embed_text(embedding_text)

        # ── BUILD PAYLOAD ─────────────────────────────────────
        # Payload = Qdrant's term for metadata stored with a point.
        # Used for filtering before similarity search.
        # Same concept as ChromaDB metadata.
        #
        # Qdrant payload supports: str, int, float, bool, list, dict
        # More flexible than ChromaDB (which required flat str/int/float/bool)
        payload = {
            "event_type":       event.event_type.value,
            "emotional_states": [e.value for e in event.emotional_states],
            "has_people":       len(event.people) > 0,
            "confidence":       event.confidence_score or 0,
            "event_date":       event.event_date or datetime.now().strftime("%Y-%m-%d"),
            "tags":             event.tags,
            "embedding_text":   embedding_text,  # store for debugging
        }

        # ── UPSERT INTO QDRANT ────────────────────────────────
        # PointStruct = one point to store
        #   id:      must be UUID or integer in Qdrant
        #   vector:  the embedding
        #   payload: metadata dict
        #
        # upsert = insert if new, update if ID exists
        # We pass event_id as the point ID — same as ChromaDB approach.
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id      = event_id,   # Qdrant accepts UUID strings
                    vector  = vector,
                    payload = payload
                )
            ]
        )

        count = self.client.count(self.collection_name).count
        print(f"   🔮 Vector stored: event {event_id[:8]}... ({count} total vectors)")

        return event_id

    def search_similar(self,
                       query_text: str,
                       n_results: int = 5,
                       filter_event_type: str = None,
                       filter_emotion: str = None) -> list[dict]:
        """
        Find events semantically similar to a query.

        WHAT CHANGED FROM CHROMADB VERSION:
        1. query_vector passed directly (not wrapped in list)
        2. Filter uses Qdrant's Filter/FieldCondition objects
        3. score in results IS similarity (not distance)
           So NO "1 - distance" conversion needed here

        INTERNAL — Qdrant filter syntax:
        Filter(must=[...]) = AND conditions
        FieldCondition(key, match=MatchValue(value)) = equality check
        This is more verbose than ChromaDB's dict syntax but more powerful.
        """

        # ── EMBED QUERY ───────────────────────────────────────
        query_vector = self.embedding_engine.embed_text(query_text)

        # ── BUILD FILTER ──────────────────────────────────────
        filter_conditions = []

        if filter_event_type:
            filter_conditions.append(
                FieldCondition(
                    key   = "event_type",
                    match = MatchValue(value=filter_event_type)
                )
            )

        if filter_emotion:
            # Qdrant supports filtering on array fields natively
            # MatchValue on a list field checks if the value is in the list
            filter_conditions.append(
                FieldCondition(
                    key   = "emotional_states",
                    match = MatchValue(value=filter_emotion)
                )
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # ── GET TOTAL COUNT FOR LIMIT ─────────────────────────
        total = self.client.count(self.collection_name).count
        if total == 0:
            return []

        # ── SEARCH ────────────────────────────────────────────
        from qdrant_client.models import QueryRequest

        results = self.client.query_points(
            collection_name = self.collection_name,
            query           = query_vector,
            limit           = min(n_results, total),
            query_filter    = query_filter,
            with_payload    = True
).points

        # ── FORMAT RESULTS ────────────────────────────────────
        # ScoredPoint fields:
        #   .id:      the point ID (our event_id)
        #   .score:   cosine similarity (0.0 to 1.0, higher = more similar)
        #   .payload: the metadata dict we stored
        #
        # NOTE: Qdrant score IS similarity, NOT distance.
        # No "1 - distance" conversion needed. This is cleaner than ChromaDB.
        formatted = []
        for point in results:
            formatted.append({
                "event_id":        str(point.id),
                "similarity_score": round(point.score, 4),
                "embedding_text":  point.payload.get("embedding_text", ""),
                "metadata":        point.payload
            })

        print(f"   🔍 Found {len(formatted)} similar events for: '{query_text[:50]}...'")
        for r in formatted:
            print(f"      → {r['event_id'][:8]}... similarity: {r['similarity_score']:.3f}")

        return formatted

    def get_collection_stats(self) -> dict:
        """Summary of what's in the vector store."""
        count = self.client.count(self.collection_name).count
        return {
            "total_vectors":   count,
            "collection_name": self.collection_name,
        }