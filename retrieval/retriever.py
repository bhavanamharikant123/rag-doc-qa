"""Vector, BM25 and hybrid retrieval over a Chroma collection."""
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ingestion.indexer import get_vectorstore

MODES = ("vector", "hybrid")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge several ranked lists. score(doc) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            cid = doc.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_id[cid] for cid in ordered]


class Retriever:
    def __init__(self, collection: str, mode: str = "hybrid"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.store = get_vectorstore(collection)
        if self.store._collection.count() == 0:
            raise RuntimeError(
                f"Collection '{collection}' is empty. Run scripts/ingest.py first "
                f"(list collections with: python scripts/query_index.py --list)"
            )
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []
        if mode == "hybrid":
            self._build_bm25()

    def _build_bm25(self) -> None:
        data = self.store.get(include=["documents", "metadatas"])
        self._docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._docs])

    def _vector_search(self, query: str, k: int) -> list[Document]:
        return self.store.similarity_search(query, k=k)

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        tokens = _tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in top if scores[i] > 0]

    def retrieve(self, query: str, k: int = 20) -> list[Document]:
        vector_hits = self._vector_search(query, k)
        if self.mode == "vector":
            return vector_hits
        bm25_hits = self._bm25_search(query, k)
        return reciprocal_rank_fusion([vector_hits, bm25_hits])[:k]