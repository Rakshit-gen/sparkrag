from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from sparkrag.config import CHROMA_DIR, EMBEDDING_MODEL


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def write_chunks(rows: list[dict]) -> int:
    """Upsert already-embedded chunks into Chroma.

    Chroma's add_texts recomputes embeddings itself, which would throw
    away the work Spark just did, so this goes through the lower-level
    collection API to store the vectors we already have.
    """
    store = get_vectorstore()
    docs = [
        Document(page_content=row["chunk"], metadata={"source": row["source"]})
        for row in rows
    ]
    ids = [f"{row['source']}::{row['chunk_id']}" for row in rows]
    embeddings = [row["embedding"] for row in rows]

    store._collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
    )
    return len(rows)
