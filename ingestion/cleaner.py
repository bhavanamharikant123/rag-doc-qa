"""Clean raw page text: page numbers, repeated headers/footers, hyphenation."""
import re
from collections import Counter, defaultdict

from langchain_core.documents import Document

PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.I)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _normalize(line: str) -> str:
    """'Page 3' and 'Page 17' both become 'page #' so they match each other."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_repeated_edge_lines(
    page_texts: list[str], edge: int = 3, threshold: float = 0.4
) -> set[str]:
    """Lines near the top/bottom of >= threshold of pages are headers/footers."""
    n = len(page_texts)
    if n < 4:
        return set()
    counter: Counter = Counter()
    for text in page_texts:
        lines = [l for l in text.splitlines() if l.strip()]
        edges = {_normalize(l) for l in lines[:edge] + lines[-edge:]}
        counter.update(edges)
    return {line for line, count in counter.items() if count / n >= threshold}


def clean_text(text: str, repeated: set[str] | None = None) -> str:
    repeated = repeated or set()
    text = CONTROL_CHARS_RE.sub("", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated words

    kept = []
    for line in text.splitlines():
        if PAGE_NUM_RE.match(line):
            continue
        if _normalize(line) in repeated:
            continue
        kept.append(line.rstrip())

    text = "\n".join(kept)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(docs: list[Document], min_chars: int = 30) -> list[Document]:
    by_source: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        by_source[d.metadata["source"]].append(d)

    cleaned: list[Document] = []
    for pages in by_source.values():
        pages.sort(key=lambda d: d.metadata["page"])
        repeated = find_repeated_edge_lines([p.page_content for p in pages])
        for p in pages:
            text = clean_text(p.page_content, repeated)
            if len(text) >= min_chars:
                cleaned.append(Document(page_content=text, metadata=dict(p.metadata)))
    return cleaned