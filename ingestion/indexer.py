"""Embed chunks with OpenAI and store them in a persistent Chroma collection."""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from core.config import settings


def build_collection_name(strategy: str, chunk_size: int, overlap: int) -> str:
    base = settings.collection_name
    if strategy == "semantic":
        return f"{base}_semantic"
    return f"{base}_{strategy}_{chunk_size}_{overlap}"


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_path),
    )


def index_chunks(
    chunks: list[Document], collection_name: str, reset: bool = False, batch_size: int = 100
) -> Chroma:
    store = get_vectorstore(collection_name)
    if reset:
        store.delete_collection()
        store = get_vectorstore(collection_name)

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding + indexing"):
        batch = chunks[i : i + batch_size]
        store.add_documents(documents=batch, ids=[c.metadata["chunk_id"] for c in batch])
    return store