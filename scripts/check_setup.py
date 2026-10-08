"""Sanity check: confirms the key, LLM, embeddings, and Chroma all work."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from core.config import settings


def main():
    print("1) Testing LLM ...")
    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    reply = llm.invoke("Reply with exactly: LLM OK")
    print("   ->", reply.content)

    print("2) Testing embeddings ...")
    emb = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    vec = emb.embed_query("hello world")
    print(f"   -> embedding dimension: {len(vec)}")

    print("3) Testing Chroma ...")
    store = Chroma(
        collection_name="setup_test",
        embedding_function=emb,
        persist_directory=str(settings.chroma_path / "_test"),
    )
    store.add_texts(
        ["Paris is the capital of France.", "Bananas are yellow."],
        metadatas=[{"source": "a"}, {"source": "b"}],
    )
    hits = store.similarity_search("What is the capital of France?", k=1)
    print("   ->", hits[0].page_content)

    print("\nAll checks passed. Setup is complete.")


if __name__ == "__main__":
    main()