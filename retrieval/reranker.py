"""Cross-encoder reranking with sentence-transformers."""
from langchain_core.documents import Document

# Good quality, ~1 GB download on first use:
DEFAULT_MODEL = "BAAI/bge-reranker-base"
# Much smaller (~90 MB) and faster, slightly less accurate:
# DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import CrossEncoder  # lazy: slow import

        print(f"Loading reranker '{model_name}' (first run downloads the model)...")
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
        if not docs:
            return []
        scores = self.model.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        out = []
        for doc, score in ranked[:top_n]:
            doc.metadata["rerank_score"] = float(score)
            out.append(doc)
        return out