"""
rag.py - Retrieval-Augmented Generation with a graceful backend fallback.

Two backends, same public API (`RagIndex.build()`, `.search()`, `.stats()`):

  1) Embeddings backend  (preferred)
     - Uses sentence-transformers ("all-MiniLM-L6-v2") for semantic embeddings.
     - Stored in a persistent ChromaDB collection under ~/.smart_chatbot/chroma/.
     - Understands meaning (e.g. "who is a student" matches "person admitted to a
       course") - much better than keyword search.

  2) TF-IDF backend  (fallback)
     - Pure-Python, no downloads, no extra deps. Always available.
     - Used when chromadb or sentence-transformers are not installed.

Chunks come from `chunk_text()` which is shared by both backends.
"""

from __future__ import annotations
import os
import re
import math
import hashlib
from collections import Counter


# ---------------------------------------------------------------------------
# Chunking (shared by both backends)
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = 800, overlap: int = 250):
    """Split a long string into overlapping character-based chunks."""
    text = (text or "").strip()
    chunks = []
    i = 0
    step = max(size - overlap, 1)
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append(piece)
        i += step
    return chunks


# ---------------------------------------------------------------------------
# Backend availability check
# ---------------------------------------------------------------------------
def _embeddings_available():
    """Return True only when chromadb and sentence-transformers can be imported."""
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


HAS_EMBEDDINGS = _embeddings_available()


# ---------------------------------------------------------------------------
# TF-IDF backend (fallback)
# ---------------------------------------------------------------------------
def _stem(w: str) -> str:
    """Very light stemmer that strips common suffixes to improve recall."""
    for suf in ("ements", "ement", "ations", "ation", "ings", "ing",
                "ment", "ies", "ied", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def _tokenize(text: str):
    """Lowercase, split on non-alphanumerics, and stem each token."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


class _TfidfIndex:
    """TF-IDF over chunks with cosine similarity."""

    backend = "tfidf"

    def __init__(self):
        self.chunks = []
        self.tfs = []
        self.idf = {}

    def build(self, docs):
        self.chunks, self.tfs = [], []
        for d in docs:
            for ch in chunk_text(d.get("text", "")):
                self.chunks.append({"source": d.get("name", "doc"), "text": ch})
                self.tfs.append(Counter(_tokenize(ch)))
        n = len(self.tfs) or 1
        df = Counter()
        for tf in self.tfs:
            for term in tf:
                df[term] += 1
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def _vec(self, tf):
        return {t: f * self.idf.get(t, 0.0) for t, f in tf.items()}

    def search(self, query: str, k: int = 4):
        if not self.chunks:
            return []
        qv = self._vec(Counter(_tokenize(query)))
        qnorm = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        scored = []
        for i, tf in enumerate(self.tfs):
            dv = self._vec(tf)
            dnorm = math.sqrt(sum(v * v for v in dv.values())) or 1.0
            dot = sum(qv.get(t, 0.0) * v for t, v in dv.items())
            scored.append((dot / (qnorm * dnorm), i))
        scored.sort(reverse=True)
        out = []
        for s, i in scored[:k]:
            if s > 0:
                out.append({"source": self.chunks[i]["source"],
                            "text": self.chunks[i]["text"],
                            "score": round(s, 3)})
        return out

    def stats(self):
        sources = sorted({c["source"] for c in self.chunks})
        return {"chunks": len(self.chunks), "sources": sources, "backend": self.backend}


# ---------------------------------------------------------------------------
# Embeddings backend (preferred)
# ---------------------------------------------------------------------------
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_CHROMA_DIR = os.path.join(os.path.expanduser("~"), ".smart_chatbot", "chroma")

# Cache the embedding model + client at module level so we only load once
# (loading takes ~1-2 seconds and 90 MB of RAM).
_embed_model = None
_chroma_client = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        os.makedirs(_CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)
    return _chroma_client


class _EmbeddingsIndex:
    """Semantic search backend using sentence-transformers + ChromaDB."""

    backend = "embeddings"

    def __init__(self, collection_name="kb"):
        self.collection_name = collection_name
        self._sources = []
        self._count = 0
        self._collection = None

    def _collection_handle(self, reset=False):
        client = _get_chroma_client()
        if reset:
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, docs):
        """(Re)index the given docs. Old collection is dropped and rebuilt."""
        model = _get_embed_model()
        col = self._collection_handle(reset=True)

        all_texts, all_meta, all_ids = [], [], []
        for d in docs:
            name = d.get("name", "doc")
            for i, ch in enumerate(chunk_text(d.get("text", ""))):
                # deterministic id: source name + chunk index + short content hash
                h = hashlib.md5(ch.encode("utf-8", errors="ignore")).hexdigest()[:8]
                all_ids.append(f"{name}::{i}::{h}")
                all_texts.append(ch)
                all_meta.append({"source": name})

        if not all_texts:
            self._collection = col
            self._sources, self._count = [], 0
            return

        # Batch-encode all chunks (fast; ~200 chunks/second on CPU)
        embeddings = model.encode(all_texts, show_progress_bar=False,
                                  convert_to_numpy=True).tolist()
        # Chroma has a per-batch limit; add in slices to be safe
        BATCH = 500
        for j in range(0, len(all_texts), BATCH):
            col.add(
                ids=all_ids[j:j + BATCH],
                documents=all_texts[j:j + BATCH],
                metadatas=all_meta[j:j + BATCH],
                embeddings=embeddings[j:j + BATCH],
            )
        self._collection = col
        self._sources = sorted({m["source"] for m in all_meta})
        self._count = len(all_texts)

    def search(self, query: str, k: int = 4):
        if self._collection is None or self._count == 0:
            return []
        model = _get_embed_model()
        q_emb = model.encode([query], convert_to_numpy=True).tolist()
        res = self._collection.query(query_embeddings=q_emb, n_results=k)
        out = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            # Chroma returns cosine DISTANCE (0 = identical, 2 = opposite);
            # convert to similarity in [0, 1] for easier reasoning.
            score = round(max(0.0, 1.0 - float(dist)), 3)
            if score > 0:
                out.append({"source": meta.get("source", "doc"),
                            "text": text, "score": score})
        return out

    def stats(self):
        return {"chunks": self._count, "sources": self._sources,
                "backend": self.backend}


# ---------------------------------------------------------------------------
# Public facade - picks the best available backend automatically.
# ---------------------------------------------------------------------------
class RagIndex:
    """Automatically picks the embeddings backend when available,
    otherwise falls back to TF-IDF. Interface is identical either way."""

    def __init__(self, collection_name: str = "kb"):
        if HAS_EMBEDDINGS:
            try:
                # Pre-load the model so a download failure surfaces here
                # (before build/search) and we can fall back cleanly.
                _get_embed_model()
                self._impl = _EmbeddingsIndex(collection_name=collection_name)
            except Exception:
                self._impl = _TfidfIndex()
        else:
            self._impl = _TfidfIndex()

    def build(self, docs):
        try:
            self._impl.build(docs)
        except Exception:
            # If embeddings backend crashes mid-build, fall back and retry once
            if not isinstance(self._impl, _TfidfIndex):
                self._impl = _TfidfIndex()
                self._impl.build(docs)
            else:
                raise

    def search(self, query, k=4):
        return self._impl.search(query, k=k)

    def stats(self):
        return self._impl.stats()

    @property
    def chunks(self):
        # Kept for compatibility with app.py's task-broad-context code path
        # (used by TF-IDF backend; embeddings path uses .search only).
        return getattr(self._impl, "chunks", [])
