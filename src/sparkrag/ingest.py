from sparkrag.chunker import chunk_documents
from sparkrag.embedder import embed_chunks
from sparkrag.loader import load_documents
from sparkrag.spark_session import get_spark
from sparkrag.store import write_chunks


def run_ingest(input_dir: str) -> int:
    spark = get_spark("sparkrag-ingest")
    try:
        docs = load_documents(spark, input_dir)
        chunks = chunk_documents(docs)
        embedded = embed_chunks(chunks)
        rows = [row.asDict() for row in embedded.collect()]
    finally:
        spark.stop()

    return write_chunks(rows)
