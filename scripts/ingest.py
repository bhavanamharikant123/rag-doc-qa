"""Run the full ingestion pipeline: load -> clean -> chunk -> embed -> index."""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tiktoken

from core.config import BASE_DIR, settings
from ingestion.chunker import STRATEGIES, chunk_documents
from ingestion.cleaner import clean_documents
from ingestion.indexer import build_collection_name, index_chunks
from ingestion.loader import load_directory

PRICE_PER_MILLION_TOKENS = 0.02  # text-embedding-3-small


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma")
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--tables", action="store_true", help="extract tables as markdown")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    parser.add_argument("--dry-run", action="store_true", help="chunk only, no embedding")
    args = parser.parse_args()

    collection = build_collection_name(args.strategy, args.chunk_size, args.overlap)
    print(f"Strategy: {args.strategy} | collection: {collection}\n")

    print("[1/4] Loading PDFs")
    raw = load_directory(settings.raw_data_path, extract_tables=args.tables)
    print(f"  -> {len(raw)} pages\n")

    print("[2/4] Cleaning")
    docs = clean_documents(raw)
    print(f"  -> {len(docs)} pages kept\n")

    print("[3/4] Chunking")
    chunks = chunk_documents(docs, args.strategy, args.chunk_size, args.overlap)
    sizes = [len(c.page_content) for c in chunks]
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = sum(len(enc.encode(c.page_content)) for c in chunks)
    print(f"  -> {len(chunks)} chunks")
    print(
        f"  -> chars: min={min(sizes)}  avg={int(statistics.mean(sizes))}  max={max(sizes)}"
    )
    print(f"  -> ~{tokens:,} tokens, est. embedding cost ${tokens / 1e6 * PRICE_PER_MILLION_TOKENS:.4f}")

    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"chunks_{collection}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, **c.metadata}, ensure_ascii=False) + "\n")
    print(f"  -> saved to {out_file.relative_to(BASE_DIR)}\n")

    if args.dry_run:
        print("Dry run: skipping embedding.")
        return

    print("[4/4] Embedding and indexing")
    store = index_chunks(chunks, collection, reset=args.reset)
    print(f"\nDone. Collection '{collection}' now holds {store._collection.count()} chunks.")


if __name__ == "__main__":
    main()