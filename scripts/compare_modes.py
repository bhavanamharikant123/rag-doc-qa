"""Run one question through 4 configurations and compare what gets retrieved."""
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.rag_chain import RAGPipeline

CONFIGS = {
    "vector": dict(mode="vector", use_rerank=False),
    "hybrid": dict(mode="hybrid", use_rerank=False),
    "vector+rerank": dict(mode="vector", use_rerank=True),
    "hybrid+rerank": dict(mode="hybrid", use_rerank=True),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--collection", default="docs_recursive_1000_200")
    args = p.parse_args()

    for name, cfg in CONFIGS.items():
        rag = RAGPipeline(args.collection, use_rewrite=False, **cfg)
        t = time.perf_counter()
        res = rag.ask(args.question)
        elapsed = time.perf_counter() - t
        pages = ", ".join(f"{s['source']} p.{s['page']}" for s in res.sources)
        print(f"\n=== {name}  ({elapsed:.1f}s) ===")
        print(f"Retrieved: {pages}")
        print(f"Answer: {res.answer[:400]}")


if __name__ == "__main__":
    main()