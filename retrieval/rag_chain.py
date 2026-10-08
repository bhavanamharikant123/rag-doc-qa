"""End-to-end RAG pipeline: rewrite -> retrieve -> rerank -> generate with citations."""
import re
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from core.config import settings
from retrieval.prompts import ANSWER_SYSTEM_PROMPT, NO_ANSWER
from retrieval.retriever import Retriever
from retrieval.rewriter import QueryRewriter


@dataclass
class RAGResult:
    answer: str
    standalone_question: str
    sources: list[dict] = field(default_factory=list)   # chunks shown to the LLM
    contexts: list[str] = field(default_factory=list)   # raw texts (used by RAGAS in Phase 4)
    latency: dict = field(default_factory=dict)         # seconds per stage


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        blocks.append(f"[{i}] (source: {m.get('source')}, page {m.get('page')})\n{d.page_content}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(
        self,
        collection: str,
        mode: str = "hybrid",       # "vector" or "hybrid"
        use_rerank: bool = True,
        use_rewrite: bool = True,
        fetch_k: int = 20,          # candidates from first-stage retrieval
        top_n: int = 5,             # chunks passed to the LLM
    ):
        self.retriever = Retriever(collection, mode)
        self.use_rerank = use_rerank
        self.use_rewrite = use_rewrite
        self.fetch_k = fetch_k
        self.top_n = top_n
        self.rewriter = QueryRewriter() if use_rewrite else None
        self.reranker = None
        if use_rerank:
            from retrieval.reranker import Reranker

            self.reranker = Reranker()
        self.llm = ChatOpenAI(
            model=settings.llm_model, api_key=settings.openai_api_key, temperature=0
        )

    def retrieve(self, question: str) -> list[Document]:
        docs = self.retriever.retrieve(question, k=self.fetch_k)
        if self.reranker:
            return self.reranker.rerank(question, docs, self.top_n)
        return docs[: self.top_n]

    def ask(self, question: str, history: list[dict] | None = None) -> RAGResult:
        history = history or []
        lat: dict[str, float] = {}

        t = time.perf_counter()
        standalone = self.rewriter.rewrite(question, history) if self.rewriter else question
        lat["rewrite"] = time.perf_counter() - t

        t = time.perf_counter()
        docs = self.retrieve(standalone)
        lat["retrieve"] = time.perf_counter() - t

        if not docs:
            return RAGResult(NO_ANSWER, standalone, latency=lat)

        t = time.perf_counter()
        messages = [
            ("system", ANSWER_SYSTEM_PROMPT.format(context=format_context(docs))),
            ("human", standalone),
        ]
        answer = self.llm.invoke(messages).content.strip()
        lat["generate"] = time.perf_counter() - t
        lat["total"] = sum(lat.values())

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = [
            {
                "ref": i,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "section": d.metadata.get("section"),
                "cited": i in cited,
                "rerank_score": d.metadata.get("rerank_score"),
                "text": d.page_content,
            }
            for i, d in enumerate(docs, start=1)
        ]
        return RAGResult(answer, standalone, sources, [d.page_content for d in docs], lat)