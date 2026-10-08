"""Query an indexed collection to sanity-check retrieval. Use --list to see collections."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb

from core.config import settings
from ingestion.indexer import get_vectorstore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="your question")
    parser.add_argument("--collection", help="collection name")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--list", action="store_true", help="list collections")
    args = parser.parse_args()

    if args.list:
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        for c in client.list_collections():
            name = getattr(c, "name", c)
            print(f"- {name}: {client.get_collection(name).count()} chunks")
        return

    if not (args.query and args.collection):
        parser.error("provide a query and --collection (or use --list)")

    store = get_vectorstore(args.collection)
    results = store.similarity_search_with_score(args.query, k=args.k)
    for rank, (doc, score) in enumerate(results, start=1):
        m = doc.metadata
        print(f"\n#{rank}  distance={score:.4f}  {m.get('source')} p.{m.get('page')}")
        print("-" * 70)
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()