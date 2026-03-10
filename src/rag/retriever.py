# src/rag/retriever.py
"""
Vakeel AI — RAG Retriever
Loads the FAISS index and retrieves relevant legal chunks for a given query.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────
VECTOR_STORE_DIR = Path("data/vector_store")
MODEL_NAME       = "intfloat/e5-base-v2"
DEFAULT_TOP_K    = 5


class LegalRetriever:
    """
    Retrieves relevant legal sections from the FAISS index.
    
    Usage:
        retriever = LegalRetriever()
        results   = retriever.search("landlord not returning deposit")
        for r in results:
            print(r["text"])
    """

    def __init__(self, vector_store_dir: str = None):
        self.store_dir = Path(vector_store_dir or VECTOR_STORE_DIR)
        self.index     = None
        self.metadata  = []
        self.model     = None
        self.config    = {}
        self._load()

    def _load(self):
        """Load FAISS index, metadata and embedding model."""
        print("🔄 Loading Vakeel AI retriever...")

        # ── Load config
        config_path = self.store_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found at {config_path}\n"
                f"Run the Colab embedder first and download the vector store."
            )
        with open(config_path) as f:
            self.config = json.load(f)
        print(f"   Model:   {self.config['model_name']}")
        print(f"   Vectors: {self.config['total_vectors']:,}")

        # ── Load FAISS index
        index_path = self.store_dir / "vakeel_ai.index"
        self.index = faiss.read_index(str(index_path))
        print(f"   FAISS index loaded ✅")

        # ── Load metadata
        metadata_path = self.store_dir / "metadata.jsonl"
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        print(f"   Metadata loaded: {len(self.metadata):,} entries ✅")

        # ── Load embedding model
        print(f"   Loading embedding model...")
        self.model = SentenceTransformer(self.config["model_name"])
        print(f"   Embedding model loaded ✅")

        print(f"\n✅ Retriever ready!\n")

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, filters: dict = None) -> list:
        """
        Search for relevant legal chunks.

        Args:
            query:   Natural language legal question
            top_k:   Number of results to return
            filters: Optional dict to filter by source/doc_type
                     e.g. {"source": "bare_act"} or {"doc_type": "judgment"}

        Returns:
            List of dicts with text, score, source, act_name, etc.
        """
        # e5 models need "query: " prefix for questions
        query_text = f"query: {query}"

        # Generate query embedding
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Search — get more results if filtering
        fetch_k = top_k * 3 if filters else top_k
        scores, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx]

            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if meta.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            results.append({
                "score":      float(score),
                "text":       meta["text"],
                "source":     meta.get("source", "unknown"),
                "act_name":   meta.get("act_name", ""),
                "section_id": meta.get("section_id", ""),
                "title":      meta.get("title", ""),
                "doc_type":   meta.get("doc_type", ""),
                "court":      meta.get("court", ""),
                "topic":      meta.get("topic", ""),
                "chunk_id":   meta.get("chunk_id", ""),
            })

            if len(results) >= top_k:
                break

        return results

    def search_acts_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """Search only bare acts — useful for section citations."""
        return self.search(query, top_k=top_k, filters={"source": "bare_act"})

    def search_judgments_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """Search only court judgments."""
        return self.search(query, top_k=top_k, filters={"source": "court_judgment"})

    def format_context(self, results: list) -> str:
        """
        Format retrieved results into a clean context string
        to pass into the LLM prompt.
        """
        context_parts = []

        for i, r in enumerate(results, 1):
            source_label = ""
            if r["source"] == "bare_act" and r["act_name"]:
                source_label = f"{r['act_name']}"
                if r["section_id"]:
                    source_label += f" | {r['section_id']}"
            elif r["source"] == "court_judgment":
                source_label = f"Court Judgment | {r['court']}"
            else:
                source_label = r["source"]

            context_parts.append(
                f"[Source {i}: {source_label}]\n{r['text']}"
            )

        return "\n\n---\n\n".join(context_parts)


# ═══════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════

def test_retriever():
    """Test the retriever with sample legal queries."""

    retriever = LegalRetriever()

    test_queries = [
        "punishment for murder under IPC",
        "landlord not returning security deposit",
        "cheque bounce legal action section 138",
        "employer not paying salary what to do",
        "bail application for accused",
        "cyber crime complaint IT act",
        "divorce procedure Hindu Marriage Act",
        "child custody rights",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 {query}")
        print(f"{'='*60}")

        results = retriever.search(query, top_k=3)

        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] Score: {r['score']:.4f}")
            print(f"       Source: {r['source']} | {r['act_name'] or r['court']}")
            print(f"       Text:   {r['text'][:200]}...")

        # Show formatted context (what LLM will receive)
        print(f"\n  📋 Formatted context preview:")
        context = retriever.format_context(results[:2])
        print(f"  {context[:300]}...")


if __name__ == "__main__":
    test_retriever()