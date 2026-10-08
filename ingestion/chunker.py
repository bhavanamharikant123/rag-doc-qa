"""Three chunking strategies: recursive, semantic, structure-aware (headings)."""
import hashlib
import re
from collections import defaultdict

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings

STRATEGIES = ("recursive", "semantic", "structure")
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=SEPARATORS
    )


# ---------- 1) Recursive ----------
def recursive_chunks(docs: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    return _splitter(chunk_size, overlap).split_documents(docs)


# ---------- 2) Semantic ----------
def semantic_chunks(docs: list[Document], max_chars: int = 2000) -> list[Document]:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    cap = _splitter(max_chars, 200)

    chunks: list[Document] = []
    for doc in docs:  # per page, so page numbers stay accurate
        try:
            pieces = chunker.split_documents([doc])
        except Exception:  # very short pages can break the chunker
            pieces = [doc]
        for piece in pieces:
            if len(piece.page_content) > max_chars:  # keep sizes bounded
                chunks.extend(cap.split_documents([piece]))
            else:
                chunks.append(piece)
    return chunks


# ---------- 3) Structure-aware ----------
HEADING_PATTERNS = [
    re.compile(r"^\d+(\.\d+){0,3}[.)]?\s+[A-Z][^.!?]{2,80}$"),               # 1.2 Risk Factors
    re.compile(r"^(chapter|section|article|part|appendix)\s+[\w.]+.{0,70}$", re.I),
    re.compile(r"^[A-Z][A-Z0-9\s\-&,:/()]{3,80}$"),                         # ALL CAPS TITLE
]


def is_heading(line: str) -> bool:
    line = line.strip()
    if not (4 <= len(line) <= 90) or line.endswith((",", ";")):
        return False
    return any(p.match(line) for p in HEADING_PATTERNS)


def structure_chunks(docs: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    splitter = _splitter(chunk_size, overlap)

    by_source: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        by_source[d.metadata["source"]].append(d)

    chunks: list[Document] = []
    for source, pages in by_source.items():
        pages.sort(key=lambda d: d.metadata["page"])

        sections: list[tuple[str, str, int]] = []  # (heading, text, start_page)
        heading, buf, start = "Preamble", [], pages[0].metadata["page"]

        for page in pages:
            for line in page.page_content.splitlines():
                if is_heading(line):
                    if "".join(buf).strip():
                        sections.append((heading, "\n".join(buf), start))
                    heading, buf, start = line.strip(), [], page.metadata["page"]
                else:
                    buf.append(line)
        if "".join(buf).strip():
            sections.append((heading, "\n".join(buf), start))

        for heading, text, page_no in sections:
            for piece in splitter.split_text(text):
                chunks.append(
                    Document(
                        # Prefix the heading so each chunk carries its context
                        page_content=f"{heading}\n\n{piece}",
                        metadata={"source": source, "page": page_no, "section": heading},
                    )
                )
    return chunks


# ---------- Dispatcher ----------
def chunk_documents(
    docs: list[Document], strategy: str = "recursive", chunk_size: int = 1000, overlap: int = 200
) -> list[Document]:
    if strategy == "recursive":
        chunks = recursive_chunks(docs, chunk_size, overlap)
    elif strategy == "semantic":
        chunks = semantic_chunks(docs)
    elif strategy == "structure":
        chunks = structure_chunks(docs, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}")

    # Finalize: drop tiny chunks, add ids and metadata, de-duplicate
    final: list[Document] = []
    seen: set[str] = set()
    for c in chunks:
        text = c.page_content.strip()
        if len(text) < 40:
            continue
        key = f"{c.metadata.get('source')}|{c.metadata.get('page')}|{text}"
        chunk_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        c.page_content = text
        c.metadata.update({"chunk_id": chunk_id, "strategy": strategy})
        final.append(c)
    return final