"""Interactive RAG chat in the terminal."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.rag_chain import RAGPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default="docs_recursive_1000_200")
    p.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-rewrite", action="store_true")
    p.add_argument("--show-context", action="store_true", help="print retrieved chunks")
    args = p.parse_args()

    rag = RAGPipeline(
        args.collection,
        mode=args.mode,
        use_rerank=not args.no_rerank,
        use_rewrite=not args.no_rewrite,
    )
    print(
        f"\nReady. mode={args.mode} rerank={not args.no_rerank} rewrite={not args.no_rewrite}"
        "\nType a question. Commands: /clear (reset memory), exit\n"
    )

    history: list[dict] = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q == "/clear":
            history.clear()
            print("Memory cleared.\n")
            continue

        res = rag.ask(q, history)
        if res.standalone_question != q:
            print(f"  (searched for: {res.standalone_question})")
        print(f"\nAssistant: {res.answer}\n")

        for s in res.sources:
            mark = "*" if s["cited"] else " "
            print(f"  {mark}[{s['ref']}] {s['source']}, page {s['page']}")
        print(f"  (* = cited)  latency: {res.latency.get('total', 0):.2f}s\n")

        if args.show_context:
            for s in res.sources:
                print(f"--- [{s['ref']}] ---\n{s['text'][:400]}\n")

        history += [{"role": "user", "content": q}, {"role": "assistant", "content": res.answer}]


if __name__ == "__main__":
    main()