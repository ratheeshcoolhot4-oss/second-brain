# utils/embeddings.py
#
# ============================================================
# EMBEDDINGS — THE MATHEMATICAL HEART OF YOUR MEMORY SYSTEM
# ============================================================
#
# WHAT IS AN EMBEDDING?
# ---------------------
# An embedding converts text into a LIST OF NUMBERS (a vector).
# That list of numbers represents the MEANING of the text
# as a point in high-dimensional space.
#
# text-embedding-3-small produces 1536 numbers per text.
# So every piece of text you store becomes a point
# in 1536-dimensional space.
#
# WHY DOES THIS ENABLE SEMANTIC SEARCH?
# --------------------------------------
# The embedding model was trained so that:
#   - Similar meanings → nearby points in space
#   - Different meanings → distant points in space
#
# "I felt exhausted"         → [0.23, -0.87, 0.45, ...]
# "I was really tired"       → [0.24, -0.85, 0.44, ...]  ← very close!
# "I went for a run"         → [0.71,  0.32, -0.12, ...] ← far away
#
# This "closeness" is measured by COSINE SIMILARITY:
#
#   similarity = cos(θ) where θ is angle between two vectors
#
#   cos(0°)   = 1.0  → identical direction → identical meaning
#   cos(90°)  = 0.0  → perpendicular → unrelated
#   cos(180°) = -1.0 → opposite direction → opposite meaning
#
# WHY 1536 DIMENSIONS?
# --------------------
# More dimensions = more "room" to encode nuance.
# 1536 allows the model to capture:
#   - Syntactic similarity (sentence structure)
#   - Semantic similarity (meaning)
#   - Domain similarity (work vs personal)
#   - Emotional tone
#   - ... all simultaneously, in different "directions"
#
# Think of it like this: with 2 dimensions you can only
# describe a point on a flat map. With 1536 dimensions,
# you have 1536 independent "axes" of meaning.
#
# WHY text-embedding-3-small?
# ---------------------------
# - Small = cheaper ($0.02 per 1M tokens)
# - Still 1536 dimensions
# - Better than ada-002 (previous generation)
# - For personal data volume, "small" is more than enough
# - "large" model costs 5x more with marginal improvement
# ============================================================

from openai import OpenAI
import os
from dotenv import load_dotenv
import math

load_dotenv()


class EmbeddingEngine:
    """
    Converts text to vectors (embeddings) using OpenAI's model.

    This is the component that gives your app "semantic memory" —
    the ability to find similar past events even when the exact
    words are different.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # ============================================================
        # MODEL: text-embedding-3-small
        # Dimensions: 1536
        # Cost: ~$0.02 per 1 million tokens (basically free for personal use)
        #
        # For context: if you write 500 words per day for a year,
        # that's ~130,000 tokens total.
        # Cost: $0.002 — two tenths of a cent per year.
        # ============================================================
        self.model = "text-embedding-3-small"
        self.dimensions = 1536

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single text string to a vector of 1536 floats.

        WHAT HAPPENS INTERNALLY:

        1. Text is tokenized (same tokenizer as GPT models)
        2. Tokens pass through the embedding model
           (different architecture than GPT — no generation,
            just encoding)
        3. Model produces a single vector representing the
           WHOLE text's meaning
        4. Vector is normalized (length = 1.0) so cosine
           similarity is just a dot product (faster)

        Returns: list of 1536 floats between -1 and 1
        """

        # Clean the text — embeddings are sensitive to extra whitespace
        text = text.strip().replace("\n", " ")

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            # encoding_format="float" is default
            # Alternative: "base64" for bandwidth efficiency
            # We use float for readability/debugging
        )

        # ============================================================
        # WHAT response CONTAINS:
        #
        # response.data[0].embedding → the actual vector (list of floats)
        # response.data[0].index     → position if you sent multiple texts
        # response.usage.prompt_tokens → how many tokens your text used
        # response.usage.total_tokens  → same as prompt for embeddings
        # ============================================================

        vector = response.data[0].embedding

        print(f"   📐 Embedded: '{text[:50]}...' → vector of {len(vector)} dimensions")

        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in ONE API call.

        INTERNAL: OpenAI processes a batch more efficiently than
        individual calls. One network round trip instead of N.

        Max batch size: 2048 texts per call.
        For personal use, you'll never hit this limit.
        """

        cleaned = [t.strip().replace("\n", " ") for t in texts]

        response = self.client.embeddings.create(
            model=self.model,
            input=cleaned
        )

        # Results come back in the same order as input
        # Sort by index just to be safe
        sorted_data = sorted(response.data, key=lambda x: x.index)
        vectors = [item.embedding for item in sorted_data]

        print(f"   📐 Batch embedded {len(texts)} texts → {len(vectors[0])} dimensions each")

        return vectors

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate similarity between two vectors.

        Returns: float between -1.0 and 1.0
          1.0  = identical meaning
          0.0  = unrelated
         -1.0  = opposite meaning (rare in practice)

        MATH INTERNALS:

        cosine_similarity = (A · B) / (|A| × |B|)

        Where:
          A · B = dot product = sum of (a_i × b_i) for all dimensions
          |A|   = magnitude of A = sqrt(sum of a_i²)

        Since OpenAI normalizes vectors to length 1.0:
          |A| = |B| = 1.0
          cosine_similarity = A · B (just the dot product!)

        This is why normalized vectors are preferred —
        similarity = dot product, which is very fast to compute.

        We implement it manually here so you SEE the math.
        ChromaDB does this internally when you search.
        """

        # Dot product: multiply corresponding dimensions, sum them
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Magnitudes (should be ~1.0 for OpenAI vectors, but calculate anyway)
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def build_embedding_text(self, embedding_summary: str,
                             emotional_states: list[str],
                             external_pressures: list[str],
                             people: list[str]) -> str:
        """
        Build the TEXT WE ACTUALLY EMBED for an event.

        THIS IS A CRITICAL DESIGN DECISION.

        We don't embed the raw journal text directly.
        We build a structured string that emphasizes the dimensions
        that matter for SIMILARITY MATCHING.

        WHY?
        If you embed "Accepted extra work today, tired, money issues"
        and later ask "Should I take new responsibilities? I'm exhausted"

        The overlap depends on which words dominate the embedding.

        By explicitly including emotional_states and pressures,
        we ensure the embedding captures:
          - What happened (action)
          - Internal state (emotions)
          - External context (pressures)
          - Social context (people involved)

        This produces MUCH better semantic matching for our use case
        than embedding raw text alone.

        Think of it as: crafting the "filing label" not just dumping the file.
        """

        parts = [embedding_summary]


        if emotional_states:
            parts.append(f"Emotional state: {', '.join(emotional_states)}")

        if external_pressures:
            parts.append(f"External pressures: {', '.join(external_pressures)}")

        if people:
            parts.append(f"People involved: {', '.join(people)}")

        return " | ".join(parts)


# ============================================================
# DEMONSTRATION — Run directly to see embeddings in action
# Usage: python utils/embeddings.py
# ============================================================

if __name__ == "__main__":
    engine = EmbeddingEngine()

    print("\n" + "=" * 60)
    print("  EMBEDDING ENGINE — SIMILARITY DEMONSTRATION")
    print("=" * 60)

    # These are the kinds of entries you'll actually log
    texts = [
        "I accepted extra work despite being exhausted and under money pressure",
        "Took on new responsibilities even though I was tired and stressed about finances",
        "Went for a morning walk, feeling refreshed and calm",
        "Trusted X with an important project, thought they were reliable",
        "Felt pressured by finances so agreed to something I shouldn't have",
    ]

    print("\n📐 Generating embeddings for 5 texts...")
    vectors = engine.embed_batch(texts)

    print("\n\n📊 SIMILARITY MATRIX:")
    print("(1.0 = identical meaning, 0.0 = unrelated)\n")

    # Print similarity between all pairs
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = engine.cosine_similarity(vectors[i], vectors[j])
            bar = "█" * int(sim * 20)
            print(f"  Text {i + 1} ↔ Text {j + 1}: {sim:.3f} {bar}")
            print(f"    → '{texts[i][:45]}...'")
            print(f"    → '{texts[j][:45]}...'")
            print()

    print("\n💡 INSIGHT:")
    print("  Texts 1 & 2 should show HIGH similarity (~0.9)")
    print("  Text 3 (morning walk) should show LOW similarity to others")
    print("  This is how your app finds 'similar past situations'")
    print("  without any keyword matching.")