"""Load PDFs page by page with PyMuPDF, keeping source and page metadata."""
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document


def _extract_tables_markdown(page) -> str:
    """Optional: extract tables as markdown. Silently skipped if unsupported."""
    try:
        tables = page.find_tables()
        parts = [t.to_markdown() for t in tables.tables]
        return "\n\n".join(parts)
    except Exception:
        return ""


def load_pdf(path: Path, extract_tables: bool = False) -> list[Document]:
    docs: list[Document] = []
    with fitz.open(path) as pdf:
        total = len(pdf)
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True)
            if extract_tables:
                tables_md = _extract_tables_markdown(page)
                if tables_md:
                    text += "\n\n[TABLES]\n" + tables_md
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_no, "total_pages": total},
                )
            )
    if not docs:
        print(f"  [warn] No text found in {path.name} (scanned PDF? needs OCR)")
    return docs


def load_directory(directory: Path, extract_tables: bool = False) -> list[Document]:
    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {directory}")

    all_docs: list[Document] = []
    for path in pdf_paths:
        try:
            pages = load_pdf(path, extract_tables)
            print(f"  loaded {path.name}: {len(pages)} pages")
            all_docs.extend(pages)
        except Exception as exc:  # corrupt or encrypted file
            print(f"  [skip] {path.name}: {exc}")
    return all_docs